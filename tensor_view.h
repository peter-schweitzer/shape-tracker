#pragma once

#include <assert.h>
#include <string.h>

#include "num_types.h"
#include "tensor.h"
#include "voodoo.h"

typedef struct tensor_view_t {
  tensor_t* t;
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

tensor_view_t* transposed_view(tensor_t* t) {
  u64 r = t->rank;
  assert(r >= 2);

  ALLOC_PTR(tensor_view_t, v);
  v->rank = t->rank;
  memcpy(v->dims, t->dims, sizeof(t->dims));
  memset(v->offsets, 0, sizeof(v->offsets));
  memcpy(v->strides, t->strides, sizeof(t->strides));

  SWAP(v->strides[r - 2], v->strides[r - 1]);

  return v;
}

tensor_view_t* permuted_view(tensor_t* t, u64 idxs[MAX_TENSOR_DIMS]) {
  ALLOC_PTR(tensor_view_t, v);
  v->t = t;
  v->rank = t->rank;

  for (u64 dim = 0; dim < t->rank; ++dim) {
    /* Check that all axes are unique */
    for (u64 i = dim + 1; i < t->rank; ++i) assert(idxs[dim] != idxs[i]);

    v->dims[dim] = t->dims[idxs[dim]];
    v->strides[dim] = t->strides[idxs[dim]];
  }

  return v;
}

void premute_view(tensor_view_t* v, u64 idxs[MAX_TENSOR_DIMS]) {
  u64* dims = U_DIMS(0);
  u64* strides = U_DIMS(0);

  for (u64 dim = 0; dim < v->rank; ++dim) {
    /* Check that all axes are unique */
    for (u64 i = dim + 1; i < v->rank; ++i) assert(idxs[dim] != idxs[i]);

    dims[dim] = v->dims[idxs[dim]];
    strides[dim] = v->strides[idxs[dim]];
  }

  memcpy(v->dims, dims, sizeof(v->dims));
  memcpy(v->strides, strides, sizeof(v->strides));
}

void transpose_view(tensor_view_t* v) {
  u64 r = v->t->rank;
  assert(r >= 2);

  SWAP(v->strides[r - 2], v->strides[r - 1]);
}

tensor_view_t* slice(tensor_t* t, i64 froms[MAX_TENSOR_DIMS], i64 tos[MAX_TENSOR_DIMS], u64 strides[MAX_TENSOR_DIMS]) {
  ALLOC_PTR(tensor_view_t, v);
  v->t = t;
  v->rank = t->rank;

  for (u64 i = 0; i < t->rank; ++i) {
    i64 d = t->dims[i];
    u64 stride = strides[i];

    i64 from = froms[i];
    assert((from >= -d) && from < d);

    i64 to = tos[i];
    assert((to >= -d) && to < ((i64)d));

    i64 offset = (from + d) % d;
    i64 end = (to + d) % d;

    if (offset > end) stride = -stride;

    u64 span = ABS(end - offset);
    assert(span > 0);

    v->dims[i] = 1 + (span / stride);
    v->offsets[i] = offset;
    v->strides[i] = stride;
  }

  return v;
}
