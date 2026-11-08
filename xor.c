#include <stdio.h>

#define FMT_PRINT_N_EL
#define FMT_F_PREC 5
#define FMT_F_LEN 7

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
  tensor_t* r = op_matmul(x, weight);

  u64* dims_r = r->dims;
  assert(dims_r[1] == bias->dims[1]);

  for (u64 i = 0; i < dims_r[0]; ++i) {
    tensor_t* s_r = op_slice(r, I_DIMS(i, 0), I_DIMS(i, -1), U_DIMS(1, 1));
    op_add_(s_r, bias);
    destroy_tensor(s_r);
  }

  return r;
}

tensor_t* xor_forward(tensor_t** hidden, tensor_t* x, tensor_t* w1, tensor_t* b1, tensor_t* w2, tensor_t* b2) {
  (*hidden) = op_tanh_(scuffed_linear(x, w1, b1));
  return op_tanh_(scuffed_linear((*hidden), w2, b2));
}

int main(void) {
  const u64 EPOCHS = 1000;
  const f32 LR = 0.1;

  // TODO: generate own random numbers
#pragma region copied from a reference impl
  tensor_t* w1 = make_tensor_from_shape(2, U_DIMS(2, 2));
  set(w1, U_DIMS(0, 0), 0.24835708);
  set(w1, U_DIMS(0, 1), -0.06913215);
  set(w1, U_DIMS(1, 0), 0.32384427);
  set(w1, U_DIMS(1, 1), 0.76151493);

  tensor_t* b1 = make_tensor_from_shape(2, U_DIMS(1, 2));
  fill(b1, 0.0);

  tensor_t* w2 = make_tensor_from_shape(2, U_DIMS(2, 1));
  set(w2, U_DIMS(0, 0), -0.11707669);
  set(w2, U_DIMS(1, 0), -0.11706848);

  tensor_t* b2 = make_tensor_from_shape(2, U_DIMS(1, 1));
  fill(b2, 0.0);
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
    tensor_t* hidden;
    tensor_t* y_hat = xor_forward(&hidden, x, w1, b1, w2, b2);

    if (i % 500 == 0) {
      print_mat(y_hat);
      printf("loss: %6.3f\n", calc_MSE_loss(y_hat, y));
    }

#pragma region BACKWARD PASS
    // output layer gradients
    tensor_t* error_output;
    {
      tensor_t* y_hat_dv = op_tanh_dv(y_hat);
      error_output = op_mul_(op_sub(y_hat, y), y_hat_dv);

      destroy_tensor(y_hat_dv);
      destroy_tensor(y_hat);
    }

    // hidden layer gradients
    tensor_t* error_hidden;
    {
      tensor_t* w2_T = op_transpose(w2);
      tensor_t* hidden_dv = op_tanh_dv(hidden);

      error_hidden = op_mul_(op_matmul(error_output, w2_T), hidden_dv);

      destroy_tensor(hidden_dv);
      destroy_tensor(w2_T);
    }

#pragma region gradient updates
    tensor_t* dw2 = op_matmul(op_transpose_(hidden), error_output);
    destroy_tensor(hidden);

    tensor_t* db2 = make_tensor_from_shape(2, U_DIMS(1, 1));
    set(db2, U_DIMS(0, 0), op_sum(error_output));
    destroy_tensor(error_output);

    tensor_t* dw1;
    {
      tensor_t* x_T = op_transpose(x);
      dw1 = op_matmul(x_T, error_hidden);
      destroy_tensor(x_T);
    }

    tensor_t* db1 = make_tensor_from_shape(2, U_DIMS(1, 2));
    {
      tensor_t* error_hidden_0 = op_slice(error_hidden, I_DIMS(0, 0), I_DIMS(-1, 0), U_DIMS(1, 1));
      tensor_t* error_hidden_1 = op_slice(error_hidden, I_DIMS(0, 1), I_DIMS(-1, 1), U_DIMS(1, 1));

      set(db1, U_DIMS(0, 0), op_sum(error_hidden_0));
      set(db1, U_DIMS(0, 1), op_sum(error_hidden_1));

      destroy_tensor(error_hidden_1);
      destroy_tensor(error_hidden_0);
      destroy_tensor(error_hidden);
    }
#pragma endregion

    op_sub_(w2, op_mul_s_(dw2, LR));
    op_sub_(b2, op_mul_s_(db2, LR));
    op_sub_(w1, op_mul_s_(dw1, LR));
    op_sub_(b1, op_mul_s_(db1, LR));

    destroy_tensor(db1);
    destroy_tensor(dw1);
    destroy_tensor(db2);
    destroy_tensor(dw2);
#pragma endregion
  }

#pragma region FINAL RESULT
  tensor_t* hidden;
  tensor_t* y_hat = xor_forward(&hidden, x, w1, b1, w2, b2);

  print_mat_with_name(x, "input");
  print_mat_with_name(y, "expected");
  print_mat_with_name(y_hat, "output");
  printf("loss: %6.3f\n", calc_MSE_loss(y_hat, y));

  destroy_tensor(y_hat);
  destroy_tensor(hidden);
#pragma endregion

#pragma region CLENAUP
  destroy_tensor(y);
  destroy_tensor(x);
  destroy_tensor(b2);
  destroy_tensor(w2);
  destroy_tensor(b1);
  destroy_tensor(w1);
#pragma endregion

  return 0;
}
