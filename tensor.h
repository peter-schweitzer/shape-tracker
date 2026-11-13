#pragma once

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "dat_buff.h"
#include "num_types.h"
#include "rand.h"
#include "voodoo.h"
typedef struct tensor_t {
  data_buff* buff;
  // NOTE: is_view could be made into a flags field later to have less padding
  bool is_view;
  u64 n_el;
  u64 rank;
  u64 dims[MAX_TENSOR_DIMS];
  u64 offsets[MAX_TENSOR_DIMS];
  u64 strides[MAX_TENSOR_DIMS];
} tensor_t;

tensor_t* make_tensor(u64 n_el) {
  ALLOC_PTR(tensor_t, t);

  t->buff = make_dat_buff(n_el);
  t->n_el = n_el;
  t->rank = 1;
  t->dims[0] = n_el;
  t->strides[0] = 1;

  return t;
}

void destroy_tensor(tensor_t* t) {
  dec_rc(t->buff);
  free(t);
}

tensor_t* make_tensor_from_shape(u64 rank, u64 dims[MAX_TENSOR_DIMS]) {
  u64 n_el = 1;
  for (u64 i = 0; i < rank; ++i) n_el *= dims[i];

  u64 _strides[MAX_TENSOR_DIMS] = DIMS(0);
  u64 acc = 1;
  for (u64 dim = 0; dim < rank; ++dim) {
    acc *= dims[dim];
    _strides[dim] = n_el / acc;
  }
  assert(acc == n_el);

  ALLOC_PTR(tensor_t, t);

  t->buff = make_dat_buff(n_el);
  t->n_el = n_el;
  t->rank = rank;
  u64* _dims = dims;
  CPY_FIELD(t, dims);
  CPY_FIELD(t, strides);

  return t;
}

#pragma region INITIALIZER
#define INITIALIZER(_) \
  _(zeros, 0.0)        \
  _(ones, 1.0)         \
  _(rand, get_rand_f32())

#define _(name, init)                                                       \
  tensor_t* make_##name##_from_shape(u64 rank, u64 dims[MAX_TENSOR_DIMS]) { \
    tensor_t* t = make_tensor_from_shape(rank, dims);                       \
    data_buff* db = t->buff;                                                \
    for (u64 i = 0; i < db->n_el; ++i) {                                    \
      db->data[i] = init;                                                   \
    }                                                                       \
    return t;                                                               \
  }

INITIALIZER(_)
#undef _
#pragma endregion

tensor_t* clone_tensor(tensor_t* t) {
  ALLOC_PTR(tensor_t, c);
  memcpy(c, t, sizeof(tensor_t));
  c->buff = clone_data_buff(t->buff);

  return c;
}

tensor_t* copy_tensor(tensor_t* t) {
  inc_rc(t->buff);

  ALLOC_PTR(tensor_t, c);
  memcpy(c, t, sizeof(tensor_t));

  return c;
}

u64 idx(tensor_t* t, u64 idxs[MAX_TENSOR_DIMS]) {
  i64 idx = 0;
  for (u64 dim = 0; dim < t->rank; ++dim) idx += t->offsets[dim] + (t->strides[dim] * idxs[dim]);

  assert(idx >= 0 && ((u64)idx) < t->buff->n_el);

  return (u64)idx;
}

f32 get(tensor_t* t, u64 idxs[MAX_TENSOR_DIMS]) { return t->buff->data[idx(t, idxs)]; }
void set(tensor_t* t, u64 idxs[MAX_TENSOR_DIMS], f32 val) { t->buff->data[idx(t, idxs)] = val; }

void fill(tensor_t* t, f32 val) {
  u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);
  for (u64 idx = 0; idx < t->n_el; ++idx) {
    IDX2IDXS(idx, idxs, t->rank, t->dims)
    set(t, idxs, val);
  }
}

tensor_t* make_contiguous(tensor_t* t) {
  tensor_t* c = make_tensor_from_shape(t->rank, t->dims);

  // Note: memcpy-ing contiguous parts might be better
  u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);
  for (u64 idx = 0; idx < t->n_el; ++idx) {
    IDX2IDXS(idx, idxs, t->rank, t->dims)
    set(c, idxs, get(t, idxs));
  }

  return c;
}

bool same_shape(tensor_t* a, tensor_t* b) {
  u64 rank = a->rank;
  if (rank != b->rank) return false;

  u64* dims_a = a->dims;
  u64* dims_b = b->dims;
  for (u64 dim = 0; dim < rank; ++dim)
    if (dims_a[dim] != dims_b[dim]) return false;

  return true;
}

bool is_same(tensor_t* a, tensor_t* b) {
  if (!same_shape(a, b)) return false;

  u64 n = a->n_el;
  u64 rank = a->rank;

  u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);
  for (u64 idx = 0; idx < n; ++idx) {
    IDX2IDXS(idx, idxs, rank, a->dims)
    if (get(a, idxs) != get(b, idxs)) return false;
  }

  return true;
}
