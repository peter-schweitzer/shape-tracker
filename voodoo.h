/**
 * This is voodoo.
 * The question is, is this too much.
 * [...]
 * Is this too much voodoo for the next ten centuries?
 *
 * ~ Terry A. Davis
 */

#pragma once

#include "num_types.h"

#define SWAP(a, b) \
  a ^= b;          \
  b ^= a;          \
  a ^= b

#define MAX_TENSOR_DIMS 4

#define _PAD_DIMS(d1, d2, d3, d4, _...) d1, d2, d3, d4
#define DIMS(dims...) {_PAD_DIMS(dims, 0, 0, 0, 0)}
#define U_DIMS(dims...) (u64[MAX_TENSOR_DIMS]) DIMS(dims)
#define I_DIMS(dims...) (i64[MAX_TENSOR_DIMS]) DIMS(dims)

#define _VA_CNT(type, args...) sizeof((type[]){args}) / sizeof(type)

#define ALLOC_PTR(type, name) type* name = malloc(sizeof(type))

#define ABS(x) ((x) < 0 ? -(x) : (x))

#define IDX2IDXS(idx, idxs, rank, dims) \
  {                                     \
    u64 i = idx;                        \
    for (u64 o = 0; o < rank; ++o) {    \
      u64 dim = rank - o - 1;           \
      u64 d = dims[dim];                \
      idxs[dim] = i % d;                \
      i /= d;                           \
    }                                   \
  }
