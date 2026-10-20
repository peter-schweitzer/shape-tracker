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

void reshape_tensor(tensor_t* t, u64 r, const u64 dims[MAX_TENSOR_DIMS]) {
  t->rank = r;

  u64 acc = 1;
  for (u64 dim = 0; dim < r; ++dim) {
    acc *= dims[dim];
    assert(acc <= t->n_el);

    t->dims[dim] = dims[dim];
    t->strides[dim] = t->n_el / acc;
  }
  assert(acc == t->n_el);
}

tensor_t* make_tensor_from_shape(u64 rank, u64 dims[MAX_TENSOR_DIMS]) {
  u64 n_el = 1;
  for (u64 i = 0; i < rank; ++i) n_el *= dims[i];

  tensor_t* t = make_tensor(n_el);
  reshape_tensor(t, rank, dims);

  return t;
}

#define MAKE_TENSOR_FROM_DIMS(t, dims...) tensor_t* t = make_tensor_from_shape(_VA_CNT(u64, dims), U_DIMS(dims))

tensor_t* clone_tensor(tensor_t* t) {
  tensor_t* t2 = make_tensor_from_shape(t->rank, t->dims);
  memcpy(t2->dat, t->dat, sizeof(f32) * t->n_el);
  return t2;
}

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
