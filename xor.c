#include <stdio.h>

#include "fmt.h"
#include "num_types.h"
#include "ops.h"
#include "tensor.h"
#include "tensor_view.h"
#include "voodoo.h"

f32* get_n_rand_f32(u64 n) {
  assert(n > 0);

  f32* rand_f32 = malloc(sizeof(f32) * n);
  FILE* f_ptr = fopen("/dev/urandom", "rb");
  fread(rand_f32, sizeof(f32), n, f_ptr);
  fclose(f_ptr);

  for (u64 i = 0; i < n; ++i) rand_f32[i] = (f32)(((u32)(rand_f32[i])) / 0xffffffff);

  return rand_f32;
}

param_t* scuffed_linear(param_t* x, param_t* weight, param_t* bias) {
  param_t* w_T = transpose(weight);
  param_t* r = mat_mul(x, w_T);

  u64* dims_r = dims_(r);
  assert(dims_r[1] == dims_(bias)[0]);

  tensor_view_t* _s_r = slice(r->data.t, I_DIMS(0, 0), I_DIMS(0, -1), U_DIMS(1, 1));
  param_t* s_r = make_param_from_view(_s_r);

  for (u64 i = 0; i < dims_r[0]; ++i) {
    _s_r->offsets[0] = i * dims_r[1];
    add_(s_r, bias);
  }

  free(_s_r);
  return r;
}

param_t* xor_forward(param_t* x, param_t* w1, param_t* b1, param_t* w2, param_t* b2) {
  param_t* r = scuffed_linear(x, w1, b1);

  tanh_(r);

  {
    param_t* tmp = scuffed_linear(r, w2, b2);
    destroy_param(r);
    r = tmp;
  }

  tanh_(r);

  return r;
}

int main(void) {
  const u64 EPOCHS = 1;  // 2500;
  // const f32 LR = 0.1;

  // FIXME: generate own random numbers
#pragma region copied from a reference impl
  MAKE_TENSOR_FROM_DIMS(_w1, 2, 2);
  param_t* w1 = make_param_from_tensor(_w1);
  set_(w1, U_DIMS(0, 0), -0.7014);
  set_(w1, U_DIMS(0, 1), -0.6157);
  set_(w1, U_DIMS(1, 0), -0.0192);
  set_(w1, U_DIMS(1, 1), 0.7018);

  MAKE_TENSOR_FROM_DIMS(_b1, 2);
  param_t* b1 = make_param_from_tensor(_b1);
  set_(b1, U_DIMS(0), -0.6502);
  set_(b1, U_DIMS(1), 0.1630);

  MAKE_TENSOR_FROM_DIMS(_w2, 1, 2);
  param_t* w2 = make_param_from_tensor(_w2);

  set_(w2, U_DIMS(0, 0), 0.5854);
  set_(w2, U_DIMS(0, 1), 0.1786);

  MAKE_TENSOR_FROM_DIMS(_b2, 1);
  param_t* b2 = make_param_from_tensor(_b2);
  set_(b2, U_DIMS(0), 0.3463);
#pragma endregion

  MAKE_TENSOR_FROM_DIMS(_x, U_DIMS(4, 2));
  fill(_x, 0.0);

  param_t* x = make_param_from_tensor(_x);

  set_(x, U_DIMS(1, 1), 1.0);
  set_(x, U_DIMS(2, 0), 1.0);
  set_(x, U_DIMS(3, 0), 1.0);
  set_(x, U_DIMS(3, 1), 1.0);

  MAKE_TENSOR_FROM_DIMS(_y, U_DIMS(4, 1));
  fill(_y, 0.0);
  param_t* y = make_param_from_tensor(_y);

  set_(y, U_DIMS(1, 0), 1.0);
  set_(y, U_DIMS(2, 0), 1.0);

  print_mat(x);
  print_mat(y);
  for (u64 i = 0; i < EPOCHS; ++i) {
    param_t* y_hat = xor_forward(x, w1, b1, w2, b2);

    if (i % 100 == 0) print_mat(y_hat);
    // TODO: backprop / grad
  }

  return 0;
}
