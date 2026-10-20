#pragma once

#include "num_types.h"
#include "ops.h"
#include "param.h"

// Mean Squared Error
f32 calc_MSE_loss(param_t* y_hat, param_t* y) {
  param_t* d = op_sub(y_hat, y);
  op_sqr_(d);
  f32 loss = op_mean(d);

  destroy_param(d);
  return loss;
}
