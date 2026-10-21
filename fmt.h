#pragma once

#include <inttypes.h>

#include "tensor.h"
#include "tensor_view.h"

void print_tensor(tensor_t* t) {
  printf("r: %" PRIu64 "\n", t->rank);

  printf("d: { ");
  for (u64 i = 0; i < t->rank; ++i)
    printf("%s%" PRIu64, i == 0 ? "" : ", ", t->dims[i]);
  printf(" }\n");

  printf("s: { ");
  for (u64 i = 0; i < t->rank; ++i)
    printf("%s%" PRIu64, i == 0 ? "" : ", ", t->strides[i]);
  printf(" }\n\n");

  for (u64 i = 0; i < t->n_el; ++i)
    printf("%s%.1f", i == 0 ? "" : ", ", t->dat[i]);
  printf("\n\n\n");
}

void print_tensor_mat(tensor_t* t) {
  assert(t->rank == 2);

  printf("Matrix [%" PRIu64 " x %" PRIu64 "]\n\n", t->dims[0], t->dims[1]);

  u64 idx[MAX_TENSOR_DIMS] = {0, 0, 0, 0};
  for (u64 y = 0; y < t->dims[0]; ++y) {
    idx[0] = y;
    for (u64 x = 0; x < t->dims[1]; ++x) {
      idx[1] = x;
      printf("%s%7.2f", x == 0 ? "" : ", ", tensor_get(t, idx));
    }
    printf("\n");
  }

  printf("\n\n");
}

void print_view(tensor_view_t* v) {
  printf("view of %p ( ", (void*)v->t);
  for (u64 i = 0; i < v->t->rank; ++i)
    printf("%s%" PRIu64, i == 0 ? "" : ", ", v->t->dims[i]);
  printf(" )\n");

  printf("r: %" PRIu64 "\n", v->rank);

  printf("d: { ");
  for (u64 i = 0; i < v->rank; ++i)
    printf("%s%" PRIu64, i == 0 ? "" : ", ", v->dims[i]);
  printf(" }\n");

  printf("s: { ");
  for (u64 i = 0; i < v->rank; ++i)
    printf("%s%" PRIu64, i == 0 ? "" : ", ", v->strides[i]);
  printf(" }\n\n");

  u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);
  for (u64 i = 0; i < v->n_el; ++i) {
    u64 _i = i;
    for (u64 o = 0; o < v->rank; ++o) {
      u64 dim = v->rank - o - 1;
      u64 d = v->dims[dim];
      idxs[dim] = _i % d;
      _i /= d;
    }

    printf("%s%.1f", i == 0 ? "" : ", ", view_get(v, idxs));
  }

  printf("\n\n\n");
}

void print_view_mat(tensor_view_t* v) {
  assert(v->rank == 2);

  printf("Matrix [%" PRIu64 " x %" PRIu64 "]\n\n", v->dims[0], v->dims[1]);

  u64 idx[MAX_TENSOR_DIMS] = {0, 0, 0, 0};
  for (u64 y = 0; y < v->dims[0]; ++y) {
    idx[0] = y;
    for (u64 x = 0; x < v->dims[1]; ++x) {
      idx[1] = x;
      printf("%s%7.2f", x == 0 ? "" : ", ", view_get(v, idx));
    }
    printf("\n");
  }

  printf("\n\n");
}
