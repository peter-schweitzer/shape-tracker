#include "tensor.h"
#include "tensor_view.h"

void print_tensor(tensor_t* t) {
  printf("r: %ld\n", t->rank);

  printf("d: { ");
  for (u64 i = 0; i < t->rank; ++i) printf("%s%ld", i == 0 ? "" : ", ", t->dims[i]);
  printf(" }\n");

  printf("s: { ");
  for (u64 i = 0; i < t->rank; ++i) printf("%s%ld", i == 0 ? "" : ", ", t->strides[i]);
  printf(" }\n\n");

  for (u64 i = 0; i < t->n_el; ++i) printf("%s%.1f", i == 0 ? "" : ", ", t->dat[i]);
  printf("\n\n\n");
}

void print_tensor_mat(tensor_t* t) {
  assert(t->rank == 2);

  printf("Matrix [%ld x %ld]\n\n", t->dims[0], t->dims[1]);

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

void smart_print_tensor(tensor_t* t) { (t->rank == 2 ? print_tensor_mat : print_tensor)(t); }

void print_view_mat(tensor_view_t* v) {
  assert(v->rank == 2);

  printf("Matrix %ld x %ld\n\n", v->dims[0], v->dims[1]);

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
