#pragma once

#include <inttypes.h>
#include <stdio.h>

#include "tensor.h"
#include "voodoo.h"

#define FMT_F_PREC_DEFAULT 2
#define FMT_F_LEN_DEFAULT 5

#define S_I(x) #x
#define S(x) S_I(x)

#ifdef FMT_F_LEN
#define FMT_F_LEN_STR S(FMT_F_LEN)
#else
#define FMT_F_LEN_STR S(FMT_F_LEN_DEFAULT)
#endif

#ifdef FMT_F_PREC
#define FMT_F_PREC_STR S(FMT_F_PREC)
#else
#define FMT_F_PREC_STR S(FMT_F_PREC_DEFAULT)
#endif

void print_meta(tensor_t* t) {
  printf("r: %" PRIu64 "\n", t->rank);

#ifdef FMT_PRINT_N_EL
  printf("n: %" PRIu64 "\n", t->n_el);
#endif

  PP_FIELD(t, dims, d);
  PP_FIELD(t, offsets, o);
  PP_FIELD(t, strides, s);

  printf("\n");
}

void print_meta_with_name(tensor_t* t, const char* name) {
  printf("tensor \"%s\" (%s)\n", name, t->is_view ? "view" : "contiguous");
  print_meta(t);
}

void print(tensor_t* t) {
  print_meta(t);

  u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);
  for (u64 idx = 0; idx < t->n_el; ++idx) {
    IDX2IDXS(idx, idxs, t->rank, t->dims)
    printf("%s%." FMT_F_PREC_STR "f", idx == 0 ? "" : ", ", get(t, idxs));
  }
  printf("\n\n\n");
}

void print_with_name(tensor_t* t, const char* name) {
  print_meta_with_name(t, name);

  u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);
  for (u64 idx = 0; idx < t->n_el; ++idx) {
    IDX2IDXS(idx, idxs, t->rank, t->dims)
    printf("%s%." FMT_F_PREC_STR "f", idx == 0 ? "" : ", ", get(t, idxs));
  }
  printf("\n\n\n");
}

void print_mat(tensor_t* t) {
  assert(t->rank == 2);

  printf("Matrix [%" PRIu64 " x %" PRIu64 "] (%s)\n\n", t->dims[0], t->dims[1], t->is_view ? "view" : "contiguous");

  u64 idx[MAX_TENSOR_DIMS] = {0, 0, 0, 0};
  for (u64 y = 0; y < t->dims[0]; ++y) {
    idx[0] = y;
    for (u64 x = 0; x < t->dims[1]; ++x) {
      idx[1] = x;
      printf("%s%" FMT_F_LEN_STR "." FMT_F_PREC_STR "f", x == 0 ? "" : ", ", get(t, idx));
    }
    printf("\n");
  }

  printf("\n\n");
}

void print_mat_with_name(tensor_t* t, const char* name) {
  assert(t->rank == 2);

  printf("Matrix \"%s\" [%" PRIu64 " x %" PRIu64 "] (%s)\n\n", name, t->dims[0], t->dims[1], t->is_view ? "view" : "contiguous");

  u64 idx[MAX_TENSOR_DIMS] = {0, 0, 0, 0};
  for (u64 y = 0; y < t->dims[0]; ++y) {
    idx[0] = y;
    for (u64 x = 0; x < t->dims[1]; ++x) {
      idx[1] = x;
      printf("%s%" FMT_F_LEN_STR "." FMT_F_PREC_STR "f", x == 0 ? "" : ", ", get(t, idx));
    }
    printf("\n");
  }

  printf("\n\n");
}

#undef S
#undef S_I
