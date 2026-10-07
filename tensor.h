#pragma once

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "num_types.h"
#include "voodoo.h"

typedef struct tensor_t {
  f32* dat;
  u64 n_el;
  u64 rank;
  u64 dims[MAX_TENSOR_DIMS];
  u64 strides[MAX_TENSOR_DIMS];
} tensor_t;

tensor_t* make_tensor(u64 n_el) {
  f32* dat = malloc(sizeof(f32) * n_el);
  for (u64 i = 0; i < n_el; ++i) dat[i] = 1.0 * i;

  ALLOC_PTR(tensor_t, t);
  t->n_el = n_el;
  t->dat = dat;
  t->rank = 1;
  t->dims[0] = n_el;

  return t;
}

void destroy_tensor(tensor_t* t) {
  free(t->dat);
  free(t);
}

bool reshape_tensor(tensor_t* t, u64 r, const u64 d[MAX_TENSOR_DIMS]) {
  t->rank = r;

  u64 stride_acc = 1;
  for (u64 o = 1; o <= MAX_TENSOR_DIMS; ++o) {
    u64 i = MAX_TENSOR_DIMS - o;
    if (i < r) {
      t->strides[i] = stride_acc;

      i64 dim = d[i];
      stride_acc *= dim;
      t->dims[i] = dim;
    } else
      t->dims[i] = 0;
  }

  return true;
}

tensor_t* make_tensor_with_shape(u64 rank, u64 dims[MAX_TENSOR_DIMS]) {
  u64 n_el = 1;
  for (u64 i = 0; i < rank; ++i) n_el *= dims[i];

  tensor_t* t = make_tensor(n_el);
  reshape_tensor(t, rank, dims);

  return t;
}

#define MAKE_TENSOR_FROM_DIMS(t, dims...) tensor_t* t = make_tensor_with_shape(_VA_CNT(u64, dims), U_DIMS(dims));

u64 idx(tensor_t* t, u64 idxs[MAX_TENSOR_DIMS]) {
  u64 idx = 0;
  for (u64 dim = 0; dim < t->rank; ++dim) {
    u64 i = idxs[dim];
    assert(i < t->dims[dim]);
    idx += i * t->strides[dim];
  }

  assert(idx < t->n_el);
  return idx;
}

f32 tensor_get(tensor_t* t, u64 idxs[MAX_TENSOR_DIMS]) { return t->dat[idx(t, idxs)]; }
void tensor_set(tensor_t* t, u64 idxs[MAX_TENSOR_DIMS], f32 val) { t->dat[idx(t, idxs)] = val; }

void fill(tensor_t* t, f32 val) {
  for (u64 i = 0; i < t->n_el; ++i) t->dat[i] = val;
}

void tanh_tensor(tensor_t* t) {
  for (u64 i = 0; i < t->n_el; ++i) t->dat[i] = tanhf((float)(t->dat[i]));
}

tensor_t* clone_tensor(tensor_t* t) {
  tensor_t* t2 = make_tensor_with_shape(t->rank, t->dims);

  memcpy(t2->dat, t->dat, sizeof(f32) * t->n_el);

  return t2;
}

tensor_t* mat_mul(tensor_t* a, tensor_t* b) {
  assert(a->rank == 2 && b->rank == 2 && a->dims[1] == b->dims[0]);

  u64 i = a->dims[0];
  u64 j = b->dims[1];
  u64 k = a->dims[1];

  u64 dims[] = {i, j, 0, 0};
  tensor_t* out = make_tensor(i * j);
  reshape_tensor(out, 2, dims);

  u64* a_idx = U_DIMS(0);
  u64* b_idx = U_DIMS(0);
  u64* out_idx = U_DIMS(0);
  for (u64 x = 0; x < j; ++x) {
    b_idx[1] = x;
    out_idx[1] = x;
    for (u64 y = 0; y < i; ++y) {
      a_idx[0] = y;
      out_idx[0] = y;

      f32 acc = 0.0;
      for (u64 n = 0; n < k; ++n) {
        a_idx[1] = n;
        b_idx[0] = n;
        acc += tensor_get(a, a_idx) * tensor_get(b, b_idx);
      }

      tensor_set(out, out_idx, acc);
    }
  }

  return out;
}

tensor_t* transpose_tensor(tensor_t* t) {
  u64 r = t->rank;
  assert(r >= 2);

  tensor_t* t_T = clone_tensor(t);
  SWAP(t_T->strides[r - 2], t_T->strides[r - 1]);
  SWAP(t_T->dims[r - 2], t_T->dims[r - 1]);

  return t_T;
}
