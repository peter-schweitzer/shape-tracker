#pragma once

#include "num_types.h"
#include "ops.h"
#include "tensor.h"

// Mean Squared Error
f32 calc_MSE_loss(tensor_t* y_hat, tensor_t* y) {
  tensor_t* d = op_sub(y_hat, y);
  op_sqr_(d);
  f32 loss = op_mean(d);
  destroy_tensor(d);

  return loss;
}
