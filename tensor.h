#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#include "num_types.h"

#define MAX_TENSOR_DIMS 4

#define alloc_ptr(type, name) type* name = malloc(sizeof(type))

typedef struct tensor_t {
  f32* dat;
  u64 len;
  u64 rank;
  u64 dims[MAX_TENSOR_DIMS];
  u64 strides[MAX_TENSOR_DIMS];
} tensor_t;

tensor_t* make_tensor(u64 len);

void destroy_tensor(tensor_t* t);

f32 get(tensor_t* t, u64 idxs[MAX_TENSOR_DIMS]);
void set(tensor_t* t, u64 idxs[MAX_TENSOR_DIMS], f32 val);

bool reshape(tensor_t* v, u64 r, u64 d[MAX_TENSOR_DIMS]);
bool transpose(tensor_t* v);
tensor_t* mat_mul(tensor_t* a, tensor_t* b);

#define _PAD_DIMS(d1, d2, d3, d4, ...) d1, d2, d3, d4

#define DIMS(...) \
  (u64[MAX_TENSOR_DIMS]) { _PAD_DIMS(__VA_ARGS__, 0, 0, 0, 0) }

#define MAKE_TENSOR_FROM_DIMS(t, ...)                              \
  tensor_t* t;                                                     \
  {                                                                \
    u64 rank = sizeof((u64[]){__VA_ARGS__}) / sizeof(u64);         \
    u64 d[MAX_TENSOR_DIMS] = {_PAD_DIMS(__VA_ARGS__, 0, 0, 0, 0)}; \
    u64 len = 1;                                                   \
    for (u64 i = 0; i < rank; ++i) len *= d[i];                    \
    t = make_tensor(len);                                          \
    reshape(t, rank, d);                                           \
  }
