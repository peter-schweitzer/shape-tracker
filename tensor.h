#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "num_types.h"
#include "voodoo.h"

#define MAX_TENSOR_DIMS 4

#define alloc_ptr(type, name) type* name = malloc(sizeof(type))

typedef struct tensor_t {
  f32* dat;
  u64 len;
  u64 rank;
  u64 dims[MAX_TENSOR_DIMS];
  u64 strides[MAX_TENSOR_DIMS];
} tensor_t;

tensor_t* make_tensor(u64 len) {
  f32* dat = malloc(sizeof(f32) * len);
  for (u64 i = 0; i < len; ++i) dat[i] = 1.0 * (i + 1);

  alloc_ptr(tensor_t, t);
  t->len = len;
  t->dat = dat;
  t->rank = 1;
  t->dims[0] = len;

  return t;
}

void destroy_tensor(tensor_t* t) {
  free(t->dat);
  free(t);
}

bool reshape(tensor_t* v, u64 r, const u64 d[MAX_TENSOR_DIMS]) {
  v->rank = r;

  u64 stride_acc = 1;
  for (u64 o = 1; o <= MAX_TENSOR_DIMS; ++o) {
    u64 i = MAX_TENSOR_DIMS - o;
    if (i < r) {
      v->strides[i] = stride_acc;

      i64 dim = d[i];
      stride_acc *= dim;
      v->dims[i] = dim;
    } else
      v->dims[i] = 0;
  }

  return true;
}

tensor_t* make_tensor_with_shape(u64 rank, u64 dims[MAX_TENSOR_DIMS]) {
  u64 len = 1;
  for (u64 i = 0; i < rank; ++i) len *= dims[i];

  tensor_t* t = make_tensor(len);
  reshape(t, rank, dims);

  return t;
}

u64 idx(tensor_t* t, u64 idxs[MAX_TENSOR_DIMS]) {
  u64 idx = 0;
  for (u64 dim = 0; dim < t->rank; ++dim) {
    u64 i = idxs[dim];
    assert(i < t->dims[dim]);
    idx += i * t->strides[dim];
  }

  assert(idx < t->len);

  return idx;
}

f32 get(tensor_t* t, u64 idxs[MAX_TENSOR_DIMS]) { return t->dat[idx(t, idxs)]; }
void set(tensor_t* t, u64 idxs[MAX_TENSOR_DIMS], f32 val) { t->dat[idx(t, idxs)] = val; }

void fill(tensor_t* t, f32 val) {
  for (u64 i = 0; i < t->len; ++i) t->dat[i] = val;
}

bool transpose(tensor_t* t) {
  if (t->rank < 2) return false;

  i64 tmp_s = t->strides[0];
  t->strides[0] = t->strides[1];
  t->strides[1] = tmp_s;

  return true;
}

tensor_t* mat_mul(tensor_t* a, tensor_t* b) {
  if (a->rank != 2 || b->rank != 2 || a->dims[1] != b->dims[0]) return false;

  u64 i = a->dims[0];
  u64 j = b->dims[1];
  u64 k = a->dims[1];

  u64 dims[] = {i, j, 0, 0};
  tensor_t* out = make_tensor(i * j);
  reshape(out, 2, dims);

  u64 a_idx[MAX_TENSOR_DIMS] = {0, 0, 0, 0};
  u64 b_idx[MAX_TENSOR_DIMS] = {0, 0, 0, 0};
  u64 out_idx[MAX_TENSOR_DIMS] = {0, 0, 0, 0};
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
        acc += get(a, a_idx) * get(b, b_idx);
      }

      set(out, out_idx, acc);
    }
  }

  return out;
}

#define _PAD_DIMS(d1, d2, d3, d4, _...) d1, d2, d3, d4

#define DIMS(dims...) \
  (u64[MAX_TENSOR_DIMS]) { _PAD_DIMS(dims, 0, 0, 0, 0) }

#define MAKE_TENSOR_FROM_DIMS(t, dims...) \
  tensor_t* t;                            \
  {                                       \
    u64* d = (DIMS(dims));                \
    u64 rank = _VA_CNT(u64, dims);        \
    t = make_tensor_with_shape(rank, d);  \
  }
