#pragma once

#include <string.h>

#include "fmt.h"
#include "num_types.h"
#include "tensor.h"
#include "voodoo.h"

#pragma region macros
#define LOAD_TENSOR_META(p) \
  u64 numel = p->n_el;      \
  u64 rank = p->rank;       \
  u64* dims = p->dims

#define GEN_UNARY_OUT_OF_PLACE_OP(name) \
  tensor_t* name(tensor_t* p) {         \
    tensor_t* c = copy_tensor(p);       \
    name##_(c);                         \
    return c;                           \
  }

#define GEN_BINARY_OUT_OF_PLACE_OP(name)     \
  tensor_t* name(tensor_t* a, tensor_t* b) { \
    tensor_t* c = copy_tensor(a);            \
    name##_(c, b);                           \
    return c;                                \
  }
#pragma endregion

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
  void op_##name##_(tensor_t* p) {          \
    LOAD_TENSOR_META(p);                    \
                                            \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);    \
    for (u64 idx = 0; idx < numel; ++idx) { \
      IDX2IDXS(idx, idxs, rank, dims)       \
                                            \
      f32 v = get(p, idxs);                 \
      set(p, idxs, op(v));                  \
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
  void op_##name##_(tensor_t* p) {          \
    LOAD_TENSOR_META(p);                    \
                                            \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);    \
    for (u64 idx = 0; idx < numel; ++idx) { \
      IDX2IDXS(idx, idxs, rank, dims)       \
                                            \
      f32 v = get(p, idxs);                 \
      set(p, idxs, (op v));                 \
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

#define _(name, op)                             \
  void op_##name##_(tensor_t* a, tensor_t* b) { \
    assert(same_shape(a, b));                   \
                                                \
    LOAD_TENSOR_META(a);                        \
                                                \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);        \
    for (u64 idx = 0; idx < numel; ++idx) {     \
      IDX2IDXS(idx, idxs, rank, dims)           \
                                                \
      f32 v_a = get(a, idxs);                   \
      f32 v_b = get(b, idxs);                   \
      set(a, idxs, op(v_a, v_b));               \
    }                                           \
  }                                             \
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

#define _(name, op)                             \
  void op_##name##_(tensor_t* a, tensor_t* b) { \
    assert(same_shape(a, b));                   \
                                                \
    LOAD_TENSOR_META(a);                        \
                                                \
    u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);        \
    for (u64 idx = 0; idx < numel; ++idx) {     \
      IDX2IDXS(idx, idxs, rank, dims)           \
                                                \
      f32 v_a = get(a, idxs);                   \
      f32 v_b = get(b, idxs);                   \
      set(a, idxs, (v_a op v_b));               \
    }                                           \
  }                                             \
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
void op_sqr_(tensor_t* p) {
  LOAD_TENSOR_META(p);

  u64 idxs[MAX_TENSOR_DIMS] = DIMS(0);
  for (u64 idx = 0; idx < numel; ++idx) {
    IDX2IDXS(idx, idxs, rank, dims)

    f32 v = get(p, idxs);
    set(p, idxs, (v * v));
  }
}

GEN_UNARY_OUT_OF_PLACE_OP(op_sqr)

void op_transpose_(tensor_t* t) {
  u64 r = t->rank;
  assert(r >= 2);

  SWAP(t->dims[r - 2], t->dims[r - 1]);
  SWAP(t->offsets[r - 2], t->offsets[r - 1]);
  SWAP(t->strides[r - 2], t->strides[r - 1]);
}

GEN_UNARY_OUT_OF_PLACE_OP(op_transpose)
#pragma endregion

#pragma region idk slice and reshape i guess
void op_slice_(tensor_t* t, i64 froms[MAX_TENSOR_DIMS], i64 tos[MAX_TENSOR_DIMS], u64 strides[MAX_TENSOR_DIMS]) {
  u64 _n_el = 1;
  u64 _dims[MAX_TENSOR_DIMS] = DIMS(0);
  u64 _offsets[MAX_TENSOR_DIMS] = DIMS(0);
  u64 _strides[MAX_TENSOR_DIMS] = DIMS(0);

  for (u64 dim = 0; dim < t->rank; ++dim) {
    u64 d = t->dims[dim];

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

    u64 s_t = t->strides[dim];
    _offsets[dim] = t->offsets[dim] + (offset * s_t);
    _strides[dim] = stride * s_t;
  }

  t->n_el = _n_el;
  t->is_view = true;
  CPY_FIELD(t, dims);
  CPY_FIELD(t, offsets);
  CPY_FIELD(t, strides);
}

tensor_t* op_slice(tensor_t* p, i64 froms[MAX_TENSOR_DIMS], i64 tos[MAX_TENSOR_DIMS], u64 strides[MAX_TENSOR_DIMS]) {
  tensor_t* r = copy_tensor(p);
  op_slice_(r, froms, tos, strides);
  return r;
}

void op_reshape_(tensor_t* t, u64 rank, u64 dims[MAX_TENSOR_DIMS]) {
  u64 _offsets[MAX_TENSOR_DIMS] = DIMS(0);
  u64 _strides[MAX_TENSOR_DIMS] = DIMS(0);

  if (t->is_view) {
    u64 dims_t[MAX_TENSOR_DIMS] = DIMS(0);
    u64 r_t = 0;
    for (u64 dim = 0; dim < t->rank; ++dim) {
      _offsets[r_t] += t->offsets[dim];
      if (t->dims[dim] > 1) dims_t[r_t++] = t->dims[dim];
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
      _strides[dim] = t->n_el / acc;
    }
    assert(acc == t->n_el);
  }

  t->rank = rank;
  u64* _dims = dims;
  CPY_FIELD(t, dims);
  CPY_FIELD(t, offsets);
  CPY_FIELD(t, strides);
}

tensor_t* op_reshape(tensor_t* t, u64 rank, u64 dims[MAX_TENSOR_DIMS]) {
  tensor_t* r;
  u64 _offsets[MAX_TENSOR_DIMS] = DIMS(0);
  u64 _strides[MAX_TENSOR_DIMS] = DIMS(0);

  if (t->is_view) {
    u64 dims_t[MAX_TENSOR_DIMS] = DIMS(0);
    u64 r_t = 0;
    for (u64 dim = 0; dim < t->rank; ++dim) {
      _offsets[r_t] += dims_t[dim];
      if (t->dims[dim] > 1) dims_t[r_t++] = dims[dim];
    }

    if (r_t == 0) {
      r_t = 1;
      dims_t[0] = 1;
    }

    u64 dims_r[MAX_TENSOR_DIMS] = DIMS(0);
    u64 r_r = 0;
    for (u64 dim = 0; dim < rank; ++dim)
      if (t->dims[dim] > 1) dims_r[r_r++] = dims[dim];

    if (r_t != r_r) {
      r = make_contiguous(t);
      FILL_BUFF(_offsets, 0);
      goto reshape;
    }

    for (u64 dim = 0; dim < r_r; ++dim)
      if (dims_t[dim] != dims_r[dim]) {
        r = make_contiguous(t);
        FILL_BUFF(_offsets, 0);
        goto reshape;
      }
  }

  r = copy_tensor(t);

reshape:
  {
    u64 acc = 1;
    for (u64 dim = 0; dim < rank; ++dim) {
      acc *= dims[dim];
      _strides[dim] = t->n_el / acc;
    }
    assert(acc == t->n_el);
  }

  r->rank = rank;
  u64* _dims = dims;
  CPY_FIELD(r, dims);
  CPY_FIELD(r, offsets);
  CPY_FIELD(r, strides);

  return r;
}
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
