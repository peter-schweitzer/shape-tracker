#pragma once

#include <inttypes.h>

#include "tensor.h"
#include "tensor_view.h"

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
    printf("%s%.1f", idx == 0 ? "" : ", ", get(t, idxs));
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
      printf("%s%7.2f", x == 0 ? "" : ", ", get(t, idx));
    }
    printf("\n");
  }

  printf("\n\n");
}
