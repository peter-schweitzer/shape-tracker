#pragma once

#include "num_types.h"
#include "ops.h"
#include "tensor.h"

// Mean Squared Error
f32 calc_MSE_loss(tensor_t* y_hat, tensor_t* y) {
  tensor_t* delta_y = op_sub(y_hat, y);
  f32 loss = op_mean(op_sqr_(delta_y));
  destroy_tensor(delta_y);

  return loss;
}
