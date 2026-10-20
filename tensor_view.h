#pragma once

#include <assert.h>
#include <string.h>

#include "num_types.h"
#include "tensor.h"
#include "voodoo.h"

typedef struct tensor_view_t {
  tensor_t* t;
  u64 n_el;
  u64 rank;
  u64 dims[MAX_TENSOR_DIMS];
  u64 offsets[MAX_TENSOR_DIMS];
  i64 strides[MAX_TENSOR_DIMS];
} tensor_view_t;

void destroy_view(tensor_view_t* v) { free(v); }

f32 view_get(tensor_view_t* v, u64 idxs[MAX_TENSOR_DIMS]) {
  u64* idx = U_DIMS(0);
  for (u64 dim = 0; dim < v->rank; ++dim) {
    idx[dim] = v->offsets[dim] + (v->strides[dim] * idxs[dim]);
    assert(idx[dim] < v->t->dims[dim]);
  }

  return tensor_get(v->t, idx);
}

tensor_view_t* transpose_view(tensor_t* t) {
  u64 r = t->rank;
  assert(r >= 2);

  ALLOC_PTR(tensor_view_t, v);
  v->n_el = t->n_el;
  v->rank = r;
  memcpy(v->dims, t->dims, sizeof(t->dims));
  memset(v->offsets, 0, sizeof(v->offsets));
  memcpy(v->strides, t->strides, sizeof(t->strides));

  SWAP(v->strides[r - 2], v->strides[r - 1]);

  return v;
}

void transpose_view_(tensor_view_t* v) {
  u64 r = v->t->rank;
  assert(r >= 2);

  SWAP(v->dims[r - 2], v->dims[r - 1]);
  SWAP(v->offsets[r - 2], v->offsets[r - 1]);
  SWAP(v->strides[r - 2], v->strides[r - 1]);
}

tensor_view_t* permute_tensor(tensor_t* t, u64 idxs[MAX_TENSOR_DIMS]) {
  ALLOC_PTR(tensor_view_t, v);
  v->t = t;
  v->n_el = t->n_el;
  v->rank = t->rank;

  for (u64 dim = 0; dim < t->rank; ++dim) {
    /* Check that all axes are unique */
    for (u64 i = dim + 1; i < t->rank; ++i) assert(idxs[dim] != idxs[i]);

    v->dims[dim] = t->dims[idxs[dim]];
    v->strides[dim] = t->strides[idxs[dim]];
  }

  return v;
}

tensor_view_t* premute_view(tensor_view_t* v, u64 idxs[MAX_TENSOR_DIMS]) {
  u64 dims[MAX_TENSOR_DIMS] = DIMS(0);
  u64 offsets[MAX_TENSOR_DIMS] = DIMS(0);
  u64 strides[MAX_TENSOR_DIMS] = DIMS(0);

  for (u64 dim = 0; dim < v->rank; ++dim) {
    /* Check that all axes are unique */
    for (u64 i = dim + 1; i < v->rank; ++i) assert(idxs[dim] != idxs[i]);

    dims[dim] = v->dims[idxs[dim]];
    offsets[dim] = v->offsets[idxs[dim]];
    strides[dim] = v->strides[idxs[dim]];
  }

  ALLOC_PTR(tensor_view_t, r);

  memcpy(r->dims, dims, sizeof(r->dims));
  memcpy(r->offsets, offsets, sizeof(r->offsets));
  memcpy(r->strides, strides, sizeof(r->strides));

  return r;
}

void premute_view_(tensor_view_t* v, u64 idxs[MAX_TENSOR_DIMS]) {
  u64 dims[MAX_TENSOR_DIMS] = DIMS(0);
  u64 offsets[MAX_TENSOR_DIMS] = DIMS(0);
  u64 strides[MAX_TENSOR_DIMS] = DIMS(0);

  for (u64 dim = 0; dim < v->rank; ++dim) {
    /* Check that all axes are unique */
    for (u64 i = dim + 1; i < v->rank; ++i) assert(idxs[dim] != idxs[i]);

    dims[dim] = v->dims[idxs[dim]];
    offsets[dim] = v->offsets[idxs[dim]];
    strides[dim] = v->strides[idxs[dim]];
  }

  memcpy(v->dims, dims, sizeof(v->dims));
  memcpy(v->offsets, offsets, sizeof(v->offsets));
  memcpy(v->strides, strides, sizeof(v->strides));
}

tensor_view_t* slice(tensor_t* t, i64 froms[MAX_TENSOR_DIMS], i64 tos[MAX_TENSOR_DIMS], u64 strides[MAX_TENSOR_DIMS]) {
  ALLOC_PTR(tensor_view_t, v);
  v->t = t;
  v->rank = t->rank;

  v->n_el = 0;
  for (u64 dim = 0; dim < t->rank; ++dim) {
    u64 d = t->dims[dim];

    i64 from = froms[dim];
    assert(from < 0 ? (u64)(-from) <= d : (u64)(from) < d);
    u64 offset = (from + d) % d;

    i64 to = tos[dim];
    assert(to < 0 ? (u64)(-to) <= d : (u64)(to) < d);
    u64 end = ((u64)(to + d)) % d;

    i64 stride = (i64)strides[dim];
    assert(stride != 0);

    i64 span = end - offset;
    if (span < 0) {
      offset = end;
      stride = -stride;
    }

    i64 _d = span / stride;
    u64 d_s = 1 + ((u64)ABS(_d));
    v->dims[dim] = d_s;
    v->n_el += d_s;

    u64 s_t = t->strides[dim];
    v->offsets[dim] = offset * s_t;
    v->strides[dim] = stride * s_t;
  }

  return v;
}
