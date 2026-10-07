#include <stdio.h>

#include "fmt.h"
#include "num_types.h"
#include "tensor.h"
#include "tensor_view.h"

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
  tensor_t* w_T = transpose_tensor(weight);
  tensor_t* r = mat_mul(x, w_T);

  assert(r->dims[1] == bias->dims[0]);

  for (u64 i0 = 0; i0 < r->dims[0]; ++i0) {
    for (u64 i1 = 0; i1 < r->dims[1]; ++i1) {
      u64 r_idx[MAX_TENSOR_DIMS] = DIMS(i0, i1);

      f32 v_r = tensor_get(r, r_idx);
      f32 v_b = tensor_get(bias, U_DIMS(i1));

      tensor_set(r, r_idx, v_r + v_b);
    }
  }

  return r;
}

tensor_t* xor_forward(tensor_t* x, tensor_t* w1, tensor_t* b1, tensor_t* w2, tensor_t* b2) {
  tensor_t* r = scuffed_linear(x, w1, b1);

  tanh_tensor(r);

  tensor_t* tmp = scuffed_linear(r, w2, b2);
  destroy_tensor(r);
  r = tmp;

  tanh_tensor(r);

  return r;
}

int main(void) {
  const u64 EPOCHS = 1;  // 2500;
  // const f32 LR = 0.1;

  // FIXME: generate own random numbers
#pragma region copied from a reference impl
  tensor_t* w1 = make_tensor_with_shape(2, U_DIMS(2, 2));
  tensor_set(w1, U_DIMS(0, 0), -0.7014);
  tensor_set(w1, U_DIMS(0, 1), -0.6157);
  tensor_set(w1, U_DIMS(1, 0), -0.0192);
  tensor_set(w1, U_DIMS(1, 1), 0.7018);

  tensor_t* b1 = make_tensor_with_shape(1, U_DIMS(2));
  tensor_set(b1, U_DIMS(0), -0.6502);
  tensor_set(b1, U_DIMS(1), 0.1630);

  tensor_t* w2 = make_tensor_with_shape(2, U_DIMS(1, 2));
  tensor_set(w2, U_DIMS(0, 0), 0.5854);
  tensor_set(w2, U_DIMS(0, 1), 0.1786);

  tensor_t* b2 = make_tensor_with_shape(1, U_DIMS(1));
  tensor_set(b2, U_DIMS(0), 0.3463);
#pragma endregion

  tensor_t* x = make_tensor_with_shape(2, U_DIMS(4, 2));
  fill(x, 0.0);
  tensor_set(x, U_DIMS(1, 1), 1.0);
  tensor_set(x, U_DIMS(2, 0), 1.0);
  tensor_set(x, U_DIMS(3, 0), 1.0);
  tensor_set(x, U_DIMS(3, 1), 1.0);

  tensor_t* y = make_tensor_with_shape(2, U_DIMS(4, 1));
  fill(y, 0.0);
  tensor_set(y, U_DIMS(1, 0), 1.0);
  tensor_set(y, U_DIMS(2, 0), 1.0);

  print_tensor_mat(x);
  print_tensor_mat(y);
  for (u64 i = 0; i < EPOCHS; ++i) {
    tensor_t* y_hat = xor_forward(x, w1, b1, w2, b2);

    if (i % 100 == 0) print_tensor_mat(y_hat);
    // TODO: backprop / grad
  }

  return 0;
}
