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
  _(tanh, tanhf)                                          \
  _(sqrt, sqrtf)                                          \
  _(log, logf)                                            \
  _(abs, ABS)

#define _(name, op)                         \
  void name##_(param_t* p) {                \
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
  GEN_UNARY_OUT_OF_PLACE_OP(name)

UNARY_ELEMENTWISE_FUNC_OPS(_)
#undef _
#pragma endregion

#pragma region UNARY_ELEMENTWISE_OP_OPS
#define UNARY_ELEMENTWISE_OP_OPS(_) \
  /* name | op */                   \
  _(neg, -)

#define _(name, op)                         \
  void name##_(param_t* p) {                \
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
  GEN_UNARY_OUT_OF_PLACE_OP(name)

UNARY_ELEMENTWISE_OP_OPS(_);
#undef _
#pragma endregion

#pragma region BINARY_ELEMENTWISE_FUNC_OPS
#define BINARY_ELEMENTWISE_FUNC_OPS(_) \
  /* name | function */                \
  _(pow, powf)

#define _(name, op)                         \
  void name##_(param_t* a, param_t* b) {    \
    assert(same_shape(a, b));               \
                                            \
    LOAD_PARAM_META(a);                     \
                                            \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);    \
    for (u64 idx = 0; idx < numel; ++idx) { \
      IDX2IDXS(idx, idxs, rank, dims)       \
                                            \
      f32 v_a = get_(a, idxs);              \
      f32 v_b = get_(b, idxs);              \
      set_(a, idxs, op(v_a, v_b));          \
    }                                       \
  }                                         \
                                            \
  GEN_BINARY_OUT_OF_PLACE_OP(name)

BINARY_ELEMENTWISE_FUNC_OPS(_)

#undef _
#pragma endregion

#pragma region BINARY_ELEMENTWISE_OP_OPS
#define BINARY_ELEMENTWISE_OP_OPS(_) \
  /* name | op */                    \
  _(add, +)                          \
  _(sub, -)                          \
  _(mul, *)                          \
  _(div, /)

#define _(name, op)                         \
  void name##_(param_t* a, param_t* b) {    \
    assert(same_shape(a, b));               \
                                            \
    LOAD_PARAM_META(a);                     \
                                            \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);    \
    for (u64 idx = 0; idx < numel; ++idx) { \
      IDX2IDXS(idx, idxs, rank, dims)       \
                                            \
      f32 v_a = get_(a, idxs);              \
      f32 v_b = get_(b, idxs);              \
      set_(a, idxs, (v_a op v_b));          \
    }                                       \
  }                                         \
                                            \
  GEN_BINARY_OUT_OF_PLACE_OP(name)

BINARY_ELEMENTWISE_OP_OPS(_)
#undef _
#pragma endregion

#pragma region REDUCE_FUNC_OPS
#define REDUCE_FUNC_OPS(_)                  \
  /* name | function | initial acc value */ \
  _(min, fminf, INFINITY)                   \
  _(max, fmaxf, -INFINITY)

#define _(name, op, acc_init)               \
  f32 name(param_t* p) {                    \
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
  f32 name(param_t* p) {                    \
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

void sqr_(param_t* p) {
  LOAD_PARAM_META(p);

  u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);
  for (u64 idx = 0; idx < numel; ++idx) {
    IDX2IDXS(idx, idxs, rank, dims)

    f32 v = get_(p, idxs);
    set_(p, idxs, (v * v));
  }
}

void transpose_(param_t* p) {
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

GEN_UNARY_OUT_OF_PLACE_OP(sqr)
GEN_UNARY_OUT_OF_PLACE_OP(transpose)

f32 mean(param_t* p) { return sum(p) / numel_(p); }

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
