#include <stdio.h>

#define FMT_PRINT_N_EL

#include "fmt.h"
#include "loss.h"
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

tensor_t* scuffed_linear(tensor_t* x, tensor_t* weight, tensor_t* bias) {
  tensor_t* w_T = op_transpose(weight);
  tensor_t* r = op_matmul(x, w_T);

  u64* dims_r = r->dims;
  assert(dims_r[1] == bias->dims[0]);

  for (u64 i = 0; i < dims_r[0]; ++i) {
    tensor_t* s_r = op_slice(r, I_DIMS(i, 0), I_DIMS(i, -1), U_DIMS(1, 1));
    op_reshape_(s_r, 1, U_DIMS(s_r->dims[1]));
    op_add_(s_r, bias);
  }

  destroy_tensor(w_T);

  return r;
}

tensor_t* xor_forward(tensor_t* x, tensor_t* w1, tensor_t* b1, tensor_t* w2, tensor_t* b2) {
  tensor_t* r = scuffed_linear(x, w1, b1);

  op_tanh_(r);

  {
    tensor_t* tmp = scuffed_linear(r, w2, b2);
    destroy_tensor(r);
    r = tmp;
  }

  op_tanh_(r);

  return r;
}

int main(void) {
  const u64 EPOCHS = 1;  // 2500;
  // const f32 LR = 0.1;

  // FIXME: generate own random numbers
#pragma region copied from a reference impl
  tensor_t* w1 = make_tensor_from_shape(2, U_DIMS(2, 2));
  set(w1, U_DIMS(0, 0), -0.7014);
  set(w1, U_DIMS(0, 1), -0.6157);
  set(w1, U_DIMS(1, 0), -0.0192);
  set(w1, U_DIMS(1, 1), 0.7018);

  tensor_t* b1 = make_tensor_from_shape(1, U_DIMS(2));
  set(b1, U_DIMS(0), -0.6502);
  set(b1, U_DIMS(1), 0.1630);

  tensor_t* w2 = make_tensor_from_shape(2, U_DIMS(1, 2));
  set(w2, U_DIMS(0, 0), 0.5854);
  set(w2, U_DIMS(0, 1), 0.1786);

  tensor_t* b2 = make_tensor_from_shape(1, U_DIMS(1));
  set(b2, U_DIMS(0), 0.3463);
#pragma endregion

#pragma region input x
  tensor_t* x = make_tensor_from_shape(2, U_DIMS(4, 2));
  fill(x, 0.0);
  set(x, U_DIMS(1, 1), 1.0);
  set(x, U_DIMS(2, 0), 1.0);
  set(x, U_DIMS(3, 0), 1.0);
  set(x, U_DIMS(3, 1), 1.0);
#pragma endregion

#pragma region expected output y
  tensor_t* y = make_tensor_from_shape(2, U_DIMS(4, 1));
  fill(y, 0.0);
  set(y, U_DIMS(1, 0), 1.0);
  set(y, U_DIMS(2, 0), 1.0);
#pragma endregion

  for (u64 i = 0; i < EPOCHS; ++i) {
    tensor_t* y_hat = xor_forward(x, w1, b1, w2, b2);

    // if (i % 100 == 0) {
    //   print_mat(y_hat);
    //   printf("loss: %6.3f\n", calc_MSE_loss(y_hat, y));
    // }
    // TODO: backprop / grad

    destroy_tensor(y_hat);
  }

  tensor_t* y_hat = xor_forward(x, w1, b1, w2, b2);

  print_mat(x);
  print_mat(y);
  print_mat(y_hat);
  printf("loss: %6.3f\n", calc_MSE_loss(y_hat, y));

  destroy_tensor(y_hat);
  destroy_tensor(y);
  destroy_tensor(x);
  destroy_tensor(b2);
  destroy_tensor(w2);
  destroy_tensor(b1);
  destroy_tensor(w1);

  return 0;
}
