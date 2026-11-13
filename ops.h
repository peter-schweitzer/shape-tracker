#pragma once

#include <assert.h>
#include <math.h>

#include "fmt.h"
#include "num_types.h"
#include "tensor.h"
#include "voodoo.h"

#define SQRT_2_DIV_PI 0.79788456080286535587989211986876f  // 0.797884583473205566f
#define GELU_COEFF 0.044715f

#pragma region macros
#define LOAD_FIELD(p, field) typeof(_Generic((p)->field, u64*: (u64*)(p)->field, default: (p)->field)) field = (p)->field

#define LOAD_TENSOR_META(p) \
  u64 numel = p->n_el;      \
  LOAD_FIELD(p, rank);      \
  LOAD_FIELD(p, dims)

#define GEN_UNARY_OUT_OF_PLACE_OP(name) \
  tensor_t* name(tensor_t* p) { return name##_(clone_tensor(p)); }

#define GEN_BINARY_OUT_OF_PLACE_OP(name) \
  tensor_t* name(tensor_t* a, tensor_t* b) { return name##_(clone_tensor(a), b); }

#define GEN_BINARY_SCALAR_OUT_OF_PLACE_OP(name) \
  tensor_t* name(tensor_t* a, f32 b) { return name##_(clone_tensor(a), b); }

#pragma endregion

#pragma region UNARY_ELEMENTWISE_FUNC_OPS
#define UNARY_ELEMENTWISE_FUNC_OPS(_) \
  /* name | op */                     \
  _(sin, sinf)                        \
  _(cos, cosf)                        \
  _(tan, tanf)                        \
  _(sinh, sinhf)                      \
  _(cosh, coshf)                      \
  _(tanh, tanhf)                      \
  _(sqrt, sqrtf)                      \
  _(log, logf)                        \
  _(round, roundf)                    \
  _(exp, expf)                        \
  _(abs, ABS)

#define _(name, op)                         \
  tensor_t* op_##name##_(tensor_t* p) {     \
    LOAD_TENSOR_META(p);                    \
                                            \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);    \
    for (u64 idx = 0; idx < numel; ++idx) { \
      IDX2IDXS(idx, idxs, rank, dims)       \
                                            \
      f32 v = get(p, idxs);                 \
      set(p, idxs, op(v));                  \
    }                                       \
                                            \
    return p;                               \
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
  tensor_t* op_##name##_(tensor_t* p) {     \
    LOAD_TENSOR_META(p);                    \
                                            \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);    \
    for (u64 idx = 0; idx < numel; ++idx) { \
      IDX2IDXS(idx, idxs, rank, dims)       \
                                            \
      f32 v = get(p, idxs);                 \
      set(p, idxs, (op v));                 \
    }                                       \
                                            \
    return p;                               \
  }                                         \
                                            \
  GEN_UNARY_OUT_OF_PLACE_OP(op_##name)

UNARY_ELEMENTWISE_OP_OPS(_)
#undef _
#pragma endregion

#pragma region UNARY_ELEMENTWISE_MACRO_OPS
#define UNARY_ELEMENTWISE_MACRO_OPS(_)                                                    \
  /* name | op */                                                                         \
  _(sqr, (v * v))                                                                         \
  _(tanh_dv, (1.0 - v * v))                                                               \
  _(step, (v >= 0.0f ? 1.0f : 0.0f))                                                      \
  _(sigmoid, (1.0f / (1.0f + expf(-v))))                                                  \
  _(hard_sigmoid, (fminf(1.0f, fmaxf(0.0f, (v + 3.0f) / 6.0f))))                          \
  _(relu, (v > 0.0f ? v : 0.0f))                                                          \
  _(silu, (v / (1.0f + expf(-v))))                                                        \
  /*GELU(x) ≈ 0.5x(1+tanh[√(2/π)(x+0.044715x^3)]) = 0.5x(1+tanh[√(2/π)x(1+0.044715x²)])*/ \
  _(gelu, (0.5f * v * (1.0f + tanhf(SQRT_2_DIV_PI * v * (1 + GELU_COEFF * v * v)))))

#define _(name, macro)                      \
  tensor_t* op_##name##_(tensor_t* p) {     \
    LOAD_TENSOR_META(p);                    \
                                            \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);    \
    for (u64 idx = 0; idx < numel; ++idx) { \
      IDX2IDXS(idx, idxs, rank, dims)       \
                                            \
      f32 v = get(p, idxs);                 \
      set(p, idxs, macro);                  \
    }                                       \
                                            \
    return p;                               \
  }                                         \
                                            \
  GEN_UNARY_OUT_OF_PLACE_OP(op_##name)

UNARY_ELEMENTWISE_MACRO_OPS(_)
#undef _
#pragma endregion

#pragma region BINARY_ELEMENTWISE_FUNC_OPS
#define BINARY_ELEMENTWISE_FUNC_OPS(_) \
  /* name | op */                      \
  _(pow, powf)

#define _(name, op)                                  \
  tensor_t* op_##name##_(tensor_t* a, tensor_t* b) { \
    assert(same_shape(a, b));                        \
                                                     \
    LOAD_TENSOR_META(a);                             \
                                                     \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);             \
    for (u64 idx = 0; idx < numel; ++idx) {          \
      IDX2IDXS(idx, idxs, rank, dims)                \
                                                     \
      f32 v_a = get(a, idxs);                        \
      f32 v_b = get(b, idxs);                        \
      set(a, idxs, op(v_a, v_b));                    \
    }                                                \
                                                     \
    return a;                                        \
  }                                                  \
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

#define _(name, op)                                  \
  tensor_t* op_##name##_(tensor_t* a, tensor_t* b) { \
    assert(same_shape(a, b));                        \
                                                     \
    LOAD_TENSOR_META(a);                             \
                                                     \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);             \
    for (u64 idx = 0; idx < numel; ++idx) {          \
      IDX2IDXS(idx, idxs, rank, dims)                \
                                                     \
      f32 v_a = get(a, idxs);                        \
      f32 v_b = get(b, idxs);                        \
      set(a, idxs, (v_a op v_b));                    \
    }                                                \
                                                     \
    return a;                                        \
  }                                                  \
                                                     \
  GEN_BINARY_OUT_OF_PLACE_OP(op_##name)

BINARY_ELEMENTWISE_OP_OPS(_)
#undef _
#pragma endregion

#pragma region BINARY_SCALAR_OP_OPS
#define BINARY_SCALAR_OP_OPS(_) \
  /* name | op */               \
  _(add_s, +)                   \
  _(sub_s, -)                   \
  _(mul_s, *)                   \
  _(div_s, /)

#define _(name, op)                            \
  tensor_t* op_##name##_(tensor_t* a, f32 b) { \
    LOAD_TENSOR_META(a);                       \
                                               \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);       \
    for (u64 idx = 0; idx < numel; ++idx) {    \
      IDX2IDXS(idx, idxs, rank, dims)          \
                                               \
      f32 v_a = get(a, idxs);                  \
      set(a, idxs, (v_a op b));                \
    }                                          \
                                               \
    return a;                                  \
  }                                            \
                                               \
  GEN_BINARY_SCALAR_OUT_OF_PLACE_OP(op_##name)

BINARY_SCALAR_OP_OPS(_)

#undef _
#pragma endregion

#pragma region REDUCE_FUNC_OPS
#define REDUCE_FUNC_OPS(_)            \
  /* name | op | initial acc value */ \
  _(min, fminf, INFINITY)             \
  _(max, fmaxf, -INFINITY)

#define _(name, op, acc_init)               \
  f32 op_##name(tensor_t* p) {              \
    LOAD_TENSOR_META(p);                    \
                                            \
    f32 acc = (acc_init);                   \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);    \
    for (u64 idx = 0; idx < numel; ++idx) { \
      IDX2IDXS(idx, idxs, rank, dims)       \
                                            \
      f32 v = get(p, idxs);                 \
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
  f32 op_##name(tensor_t* p) {              \
    LOAD_TENSOR_META(p);                    \
                                            \
    f32 acc = ((f32)(acc_init));            \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);    \
    for (u64 idx = 0; idx < numel; ++idx) { \
      IDX2IDXS(idx, idxs, rank, dims)       \
                                            \
      f32 v = get(p, idxs);                 \
      acc = (acc op v);                     \
    }                                       \
                                            \
    return acc;                             \
  }

REDUCE_OP_OPS(_)
#undef _

f32 op_mean(tensor_t* p) { return op_sum(p) / p->n_el; }
#pragma endregion

#pragma region custom inplace with gen out of place
tensor_t* op_transpose_(tensor_t* p) {
  u64 r = p->rank;
  assert(r >= 2);

  SWAP(p->dims[r - 2], p->dims[r - 1]);
  SWAP(p->offsets[r - 2], p->offsets[r - 1]);
  SWAP(p->strides[r - 2], p->strides[r - 1]);

  return p;
}

tensor_t* op_softmax_(tensor_t* p) {
  op_exp_(p);
  return op_div_s_(p, op_sum(p));
}

GEN_UNARY_OUT_OF_PLACE_OP(op_transpose)
GEN_UNARY_OUT_OF_PLACE_OP(op_softmax)
#pragma endregion

#pragma region slice, reshape and permute
tensor_t* op_slice_(tensor_t* p, i64 froms[MAX_TENSOR_DIMS], i64 tos[MAX_TENSOR_DIMS], u64 strides[MAX_TENSOR_DIMS]) {
  u64 _n_el = 1;
  u64 _dims[MAX_TENSOR_DIMS] = DIMS(0);
  u64 _offsets[MAX_TENSOR_DIMS] = DIMS(0);
  u64 _strides[MAX_TENSOR_DIMS] = DIMS(0);

  for (u64 dim = 0; dim < p->rank; ++dim) {
    u64 d = p->dims[dim];

    i64 from = froms[dim];
    assert(from < 0 ? (u64)(-from) <= d : (u64)(from) < d);
    u64 offset = (from + d) % d;

    i64 to = tos[dim];
    assert(to < 0 ? (u64)(-to) <= d : (u64)(to) < d);
    u64 end = ((u64)(to + d)) % d;

    i64 stride = (i64)strides[dim];
    assert(stride != 0);

    i64 span = end - offset;
    if (span < 0) {
      offset = end;
      stride = -stride;
    }

    i64 _d = span / stride;
    u64 d_s = 1 + ((u64)ABS(_d));
    _n_el *= d_s;
    _dims[dim] = d_s;

    u64 s_t = p->strides[dim];
    _offsets[dim] = p->offsets[dim] + (offset * s_t);
    _strides[dim] = stride * s_t;
  }

  p->n_el = _n_el;
  p->is_view = true;
  CPY_FIELD(p, dims);
  CPY_FIELD(p, offsets);
  CPY_FIELD(p, strides);

  return p;
}

tensor_t* op_slice(tensor_t* p, i64 froms[MAX_TENSOR_DIMS], i64 tos[MAX_TENSOR_DIMS], u64 strides[MAX_TENSOR_DIMS]) { return op_slice_(copy_tensor(p), froms, tos, strides); }

tensor_t* op_reshape_(tensor_t* p, u64 rank, u64 dims[MAX_TENSOR_DIMS]) {
  u64 _offsets[MAX_TENSOR_DIMS] = DIMS(0);
  u64 _strides[MAX_TENSOR_DIMS] = DIMS(0);

  if (p->is_view) {
    u64 dims_t[MAX_TENSOR_DIMS] = DIMS(0);
    u64 r_t = 0;
    for (u64 dim = 0; dim < p->rank; ++dim) {
      _offsets[r_t] += p->offsets[dim];
      if (p->dims[dim] > 1) dims_t[r_t++] = p->dims[dim];
    }

    if (r_t == 0) {
      r_t = 1;
      dims_t[0] = 1;
    }

    u64 dims_r[MAX_TENSOR_DIMS] = DIMS(0);
    u64 r_r = 0;
    for (u64 dim = 0; dim < rank; ++dim)
      if (dims[dim] > 1) dims_r[r_r++] = dims[dim];

    if (r_r == 0) {
      r_r = 1;
      dims_r[0] = 1;
    }

    assert(r_t == r_r);
    for (u64 dim = 0; dim < r_r; ++dim) assert(dims_t[dim] == dims_r[dim]);
  }

  {
    u64 acc = 1;
    for (u64 dim = 0; dim < rank; ++dim) {
      acc *= dims[dim];
      _strides[dim] = p->n_el / acc;
    }
    assert(acc == p->n_el);
  }

  p->rank = rank;
  u64* _dims = dims;
  CPY_FIELD(p, dims);
  CPY_FIELD(p, offsets);
  CPY_FIELD(p, strides);

  return p;
}

tensor_t* op_reshape(tensor_t* p, u64 rank, u64 dims[MAX_TENSOR_DIMS]) {
  tensor_t* r;
  u64 _offsets[MAX_TENSOR_DIMS] = DIMS(0);
  u64 _strides[MAX_TENSOR_DIMS] = DIMS(0);

  if (p->is_view) {
    u64 dims_t[MAX_TENSOR_DIMS] = DIMS(0);
    u64 r_t = 0;
    for (u64 dim = 0; dim < p->rank; ++dim) {
      _offsets[r_t] += dims_t[dim];
      if (p->dims[dim] > 1) dims_t[r_t++] = dims[dim];
    }

    if (r_t == 0) {
      r_t = 1;
      dims_t[0] = 1;
    }

    u64 dims_r[MAX_TENSOR_DIMS] = DIMS(0);
    u64 r_r = 0;
    for (u64 dim = 0; dim < rank; ++dim)
      if (p->dims[dim] > 1) dims_r[r_r++] = dims[dim];

    if (r_t != r_r) {
      r = make_contiguous(p);
      FILL_BUFF(_offsets, 0);
      goto reshape;
    }

    for (u64 dim = 0; dim < r_r; ++dim)
      if (dims_t[dim] != dims_r[dim]) {
        r = make_contiguous(p);
        FILL_BUFF(_offsets, 0);
        goto reshape;
      }
  }

  r = copy_tensor(p);

reshape:
  {
    u64 acc = 1;
    for (u64 dim = 0; dim < rank; ++dim) {
      acc *= dims[dim];
      _strides[dim] = p->n_el / acc;
    }
    assert(acc == p->n_el);
  }

  r->rank = rank;
  u64* _dims = dims;
  CPY_FIELD(r, dims);
  CPY_FIELD(r, offsets);
  CPY_FIELD(r, strides);

  return r;
}

tensor_t* op_permute_(tensor_t* p, u64 idxs[MAX_TENSOR_DIMS]) {
  u64 _dims[MAX_TENSOR_DIMS] = DIMS(0);
  u64 _offsets[MAX_TENSOR_DIMS] = DIMS(0);
  u64 _strides[MAX_TENSOR_DIMS] = DIMS(0);

  LOAD_FIELD(p, rank);
  LOAD_FIELD(p, dims);
  LOAD_FIELD(p, offsets);
  LOAD_FIELD(p, strides);

  for (u64 dim = 0; dim < rank; ++dim) {
    /* Check that all axes are unique */
    for (u64 i = dim + 1; i < rank; ++i) assert(idxs[dim] != idxs[i]);

    _dims[dim] = dims[idxs[dim]];
    _offsets[dim] = offsets[idxs[dim]];
    _strides[dim] = strides[idxs[dim]];
  }

  p->is_view = true;
  CPY_FIELD(p, dims);
  CPY_FIELD(p, offsets);
  CPY_FIELD(p, strides);

  return p;
}

tensor_t* op_permute(tensor_t* p, u64 idxs[MAX_TENSOR_DIMS]) { return op_permute_(clone_tensor(p), idxs); }
#pragma endregion

#pragma region out of place ops
tensor_t* op_matmul(tensor_t* a, tensor_t* b) {
  u64* dims_a = a->dims;
  u64* dims_b = b->dims;

  assert(a->rank == 2 && b->rank == 2 && dims_a[1] == dims_b[0]);

  u64 i = dims_a[0];
  u64 j = dims_b[1];
  u64 k = dims_a[1];

  MAKE_TENSOR_FROM_DIMS(out, i, j);

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

        acc += get(a, a_idx) * get(b, b_idx);
      }

      set(out, out_idx, acc);
    }
  }

  return out;
}
#pragma endregion

/**
 * TODO: impl the following ops
 *
 * SOFTMAX_DV   "softmax'"   // R = softmax'(X)
 * SIGMOID_DV   "sigmoid'"   // R = sigmoid'(X)
 * SILU_DV      "SiLU'"      // R = silu'(X)
 * RELU_DV      "ReLU'"      // R = relu'(X)
 * GELU_DV      "GeLU'"      // R = gelu'(X)
 */
