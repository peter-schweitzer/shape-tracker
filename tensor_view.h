#pragma once

#include <assert.h>
#include <stdio.h>
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

tensor_view_t* permute_view(tensor_view_t* v, u64 idxs[MAX_TENSOR_DIMS]) {
  u64 _dims[MAX_TENSOR_DIMS] = DIMS(0);
  u64 _offsets[MAX_TENSOR_DIMS] = DIMS(0);
  u64 _strides[MAX_TENSOR_DIMS] = DIMS(0);

  for (u64 dim = 0; dim < v->rank; ++dim) {
    /* Check that all axes are unique */
    for (u64 i = dim + 1; i < v->rank; ++i) assert(idxs[dim] != idxs[i]);

    _dims[dim] = v->dims[idxs[dim]];
    _offsets[dim] = v->offsets[idxs[dim]];
    _strides[dim] = v->strides[idxs[dim]];
  }

  ALLOC_PTR(tensor_view_t, r);

  CPY_FIELD(r, dims);
  CPY_FIELD(r, offsets);
  CPY_FIELD(r, strides);

  return r;
}

void permute_view_(tensor_view_t* v, u64 idxs[MAX_TENSOR_DIMS]) {
  u64 _dims[MAX_TENSOR_DIMS] = DIMS(0);
  u64 _offsets[MAX_TENSOR_DIMS] = DIMS(0);
  u64 _strides[MAX_TENSOR_DIMS] = DIMS(0);

  for (u64 dim = 0; dim < v->rank; ++dim) {
    /* Check that all axes are unique */
    for (u64 i = dim + 1; i < v->rank; ++i) assert(idxs[dim] != idxs[i]);

    _dims[dim] = v->dims[idxs[dim]];
    _offsets[dim] = v->offsets[idxs[dim]];
    _strides[dim] = v->strides[idxs[dim]];
  }

  CPY_FIELD(v, dims);
  CPY_FIELD(v, offsets);
  CPY_FIELD(v, strides);
}
