#pragma once

#include "fmt.h"
#include "num_types.h"
#include "param.h"
#include "tensor.h"
#include "tensor_view.h"
#include "voodoo.h"

#define LOAD_PARAM_META(p) \
  u64 numel = numel_(p);   \
  u64 rank = rank_(p);     \
  u64* dims = dims_(p)

#define GEN_UNARY_OUT_OF_PLACE_OP(name) \
  param_t* name(param_t* p) {           \
    param_t* r = clone(p);              \
    name##_(r);                         \
    return r;                           \
  }

#define GEN_BINARY_OUT_OF_PLACE_OP(name)  \
  param_t* name(param_t* a, param_t* b) { \
    param_t* r = clone(a);                \
    name##_(r, b);                        \
    return r;                             \
  }

#pragma region UNARY_ELEMENTWISE_FUNC_OPS
#define UNARY_ELEMENTWISE_FUNC_OPS(_)                     \
  /* name | function (or function macro like with ABS) */ \
  _(sin, sinf)                                            \
  _(cos, cosf)                                            \
  _(tan, tanf)                                            \
  _(sinh, sinhf)                                          \
  _(cosh, coshf)                                          \
  _(tanh, tanhf)                                          \
  _(sqrt, sqrtf)                                          \
  _(log, logf)                                            \
  _(abs, ABS)

#define _(name, op)                         \
  void op_##name##_(param_t* p) {           \
    LOAD_PARAM_META(p);                     \
                                            \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);    \
    for (u64 idx = 0; idx < numel; ++idx) { \
      IDX2IDXS(idx, idxs, rank, dims)       \
                                            \
      f32 v = get_(p, idxs);                \
      set_(p, idxs, op(v));                 \
    }                                       \
  }                                         \
                                            \
  GEN_UNARY_OUT_OF_PLACE_OP(op_##name)

UNARY_ELEMENTWISE_FUNC_OPS(_)
#undef _
#pragma endregion

#pragma region UNARY_ELEMENTWISE_OP_OPS
#define UNARY_ELEMENTWISE_OP_OPS(_) \
  /* name | op */                   \
  _(neg, -)

#define _(name, op)                         \
  void op_##name##_(param_t* p) {           \
    LOAD_PARAM_META(p);                     \
                                            \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);    \
    for (u64 idx = 0; idx < numel; ++idx) { \
      IDX2IDXS(idx, idxs, rank, dims)       \
                                            \
      f32 v = get_(p, idxs);                \
      set_(p, idxs, (op v));                \
    }                                       \
  }                                         \
                                            \
  GEN_UNARY_OUT_OF_PLACE_OP(op_##name)

UNARY_ELEMENTWISE_OP_OPS(_)
#undef _
#pragma endregion

#pragma region BINARY_ELEMENTWISE_FUNC_OPS
#define BINARY_ELEMENTWISE_FUNC_OPS(_) \
  /* name | function */                \
  _(pow, powf)

#define _(name, op)                           \
  void op_##name##_(param_t* a, param_t* b) { \
    assert(same_shape(a, b));                 \
                                              \
    LOAD_PARAM_META(a);                       \
                                              \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);      \
    for (u64 idx = 0; idx < numel; ++idx) {   \
      IDX2IDXS(idx, idxs, rank, dims)         \
                                              \
      f32 v_a = get_(a, idxs);                \
      f32 v_b = get_(b, idxs);                \
      set_(a, idxs, op(v_a, v_b));            \
    }                                         \
  }                                           \
                                              \
  GEN_BINARY_OUT_OF_PLACE_OP(op_##name)

BINARY_ELEMENTWISE_FUNC_OPS(_)

#undef _
#pragma endregion

#pragma region BINARY_ELEMENTWISE_OP_OPS
#define BINARY_ELEMENTWISE_OP_OPS(_) \
  /* name | op */                    \
  _(add, +)                          \
  _(sub, -)                          \
  _(mul, *)                          \
  _(divide, /)

#define _(name, op)                           \
  void op_##name##_(param_t* a, param_t* b) { \
    assert(same_shape(a, b));                 \
                                              \
    LOAD_PARAM_META(a);                       \
                                              \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);      \
    for (u64 idx = 0; idx < numel; ++idx) {   \
      IDX2IDXS(idx, idxs, rank, dims)         \
                                              \
      f32 v_a = get_(a, idxs);                \
      f32 v_b = get_(b, idxs);                \
      set_(a, idxs, (v_a op v_b));            \
    }                                         \
  }                                           \
                                              \
  GEN_BINARY_OUT_OF_PLACE_OP(op_##name)

BINARY_ELEMENTWISE_OP_OPS(_)
#undef _
#pragma endregion

#pragma region REDUCE_FUNC_OPS
#define REDUCE_FUNC_OPS(_)                  \
  /* name | function | initial acc value */ \
  _(min, fminf, INFINITY)                   \
  _(max, fmaxf, -INFINITY)

#define _(name, op, acc_init)               \
  f32 op_##name(param_t* p) {               \
    LOAD_PARAM_META(p);                     \
                                            \
    f32 acc = (acc_init);                   \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);    \
    for (u64 idx = 0; idx < numel; ++idx) { \
      IDX2IDXS(idx, idxs, rank, dims)       \
                                            \
      f32 v = get_(p, idxs);                \
      acc = op(acc, v);                     \
    }                                       \
                                            \
    return acc;                             \
  }

REDUCE_FUNC_OPS(_)
#undef _
#pragma endregion

#pragma region REDUCE_OP_OPS
#define REDUCE_OP_OPS(_)              \
  /* name | op | initial acc value */ \
  _(sum, +, 0)

#define _(name, op, acc_init)               \
  f32 op_##name(param_t* p) {               \
    LOAD_PARAM_META(p);                     \
                                            \
    f32 acc = ((f32)(acc_init));            \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);    \
    for (u64 idx = 0; idx < numel; ++idx) { \
      IDX2IDXS(idx, idxs, rank, dims)       \
                                            \
      f32 v = get_(p, idxs);                \
      acc = (acc op v);                     \
    }                                       \
                                            \
    return acc;                             \
  }

REDUCE_OP_OPS(_)
#undef _
#pragma endregion

#pragma region custom ops

void op_sqr_(param_t* p) {
  LOAD_PARAM_META(p);

  u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);
  for (u64 idx = 0; idx < numel; ++idx) {
    IDX2IDXS(idx, idxs, rank, dims)

    f32 v = get_(p, idxs);
    set_(p, idxs, (v * v));
  }
}

void op_transpose_(param_t* p) {
  u64 r = rank_(p);
  assert(r >= 2);

  u64* dims = dims_(p);
  SWAP(dims[r - 2], dims[r - 1]);

  u64* strides = strides_(p);
  SWAP(strides[r - 2], strides[r - 1]);

  if (p->type == VIEW) {
    u64* offsets = offs_(p);
    SWAP(offsets[r - 2], offsets[r - 1]);
  }
}

GEN_UNARY_OUT_OF_PLACE_OP(op_sqr)
GEN_UNARY_OUT_OF_PLACE_OP(op_transpose)

f32 op_mean(param_t* p) { return op_sum(p) / numel_(p); }

/**
 * TODO: impl the following ops
 *
 * STEP          "step"          // R = step(X)
 * SOFTMAX       "softmax"       // R = softmax(X)
 * SOFTMAX_DV    "softmax'"      // R = softmax'(X)
 * SIGMOID       "sigmoid"       // R = sigmoid(X)
 * SIGMOID_DV    "sigmoid'"      // R = sigmoid'(X)
 * HARD_SIGMOID  "hard_sigmoid"  // R = hard_sigmoid(X)
 * SILU          "SiLU"          // R = silu(X)
 * SILU_DV       "SiLU'"         // R = silu'(X)
 * RELU          "ReLU"          // R = relu(X)
 * RELU_DV       "ReLU'"         // R = relu'(X)
 * GELU          "GeLU"          // R = gelu(X)
 * GELU_DV       "GeLU'"         // R = gelu'(X)
 * ADDS          "+ξ"            // R = X+ξ
 * SUBS          "-ξ"            // R = X-ξ
 * MULS          "*ξ"            // R = X*ξ (Hadamard product)
 * DIVS          "/ξ"            // R = X/ξ
 */

param_t* op_matmul(param_t* a, param_t* b) {
  u64* dims_a = dims_(a);
  u64* dims_b = dims_(b);

  assert(rank_(a) == 2 && rank_(b) == 2 && dims_a[1] == dims_b[0]);

  u64 i = dims_a[0];
  u64 j = dims_b[1];
  u64 k = dims_a[1];

  MAKE_TENSOR_FROM_DIMS(_out, i, j);
  param_t* out = make_param_from_tensor(_out);

  u64 a_idx[MAX_TENSOR_DIMS] = DIMS(0);
  u64 b_idx[MAX_TENSOR_DIMS] = DIMS(0);
  u64 out_idx[MAX_TENSOR_DIMS] = DIMS(0);

  for (u64 x = 0; x < j; ++x) {
    b_idx[1] = x;
    out_idx[1] = x;

    for (u64 y = 0; y < i; ++y) {
      a_idx[0] = y;
      out_idx[0] = y;

      f32 acc = 0.0;
      for (u64 n = 0; n < k; ++n) {
        a_idx[1] = n;
        b_idx[0] = n;

        acc += get_(a, a_idx) * get_(b, b_idx);
      }

      set_(out, out_idx, acc);
    }
  }

  return out;
}
