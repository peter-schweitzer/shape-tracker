#pragma once

#include "tensor.h"
#include "tensor_view.h"

typedef enum PARAM_TENSOR_TYPES { TENSOR, VIEW } PARAM_TENSOR_TYPES;

typedef struct param_t {
  union {
    tensor_t* t;
    tensor_view_t* v;
  } data;
  PARAM_TENSOR_TYPES type;
} param_t;

param_t* make_param_from_tensor(tensor_t* t) {
  ALLOC_PTR(param_t, p);
  p->type = TENSOR;
  p->data.t = t;
  return p;
}

param_t* make_param_from_view(tensor_view_t* v) {
  ALLOC_PTR(param_t, p);
  p->type = VIEW;
  p->data.v = v;
  return p;
}

// this creates a new `tensor_t` to hold the underlying data
param_t* make_param_from_shape(u64 rank, u64 dims[MAX_TENSOR_DIMS]) { return make_param_from_tensor(make_tensor_from_shape(rank, dims)); }

void destroy_param(param_t* p) {
  switch (p->type) {
    case TENSOR:
      destroy_tensor(*((tensor_t**)p));
      break;
    case VIEW:
      destroy_view(*((tensor_view_t**)p));
      break;
  }

  free(p);
}

u64 numel_(param_t* p) {
  switch (p->type) {
    case TENSOR:
      return (*((tensor_t**)p))->n_el;
    case VIEW:
      return (*((tensor_view_t**)p))->n_el;
  }

  assert(false);
}

u64 rank_(param_t* p) {
  switch (p->type) {
    case TENSOR:
      return p->data.t->rank;
    case VIEW:
      return p->data.v->rank;
  }

  assert(false);
}

u64* dims_(param_t* p) {
  switch (p->type) {
    case TENSOR:
      return p->data.t->dims;
    case VIEW:
      return p->data.v->dims;
  }

  assert(false);
}

u64* offs_(param_t* p) {
  switch (p->type) {
    case TENSOR:
      return (*((tensor_t**)p))->offsets;
    case VIEW:
      return (*((tensor_view_t**)p))->offsets;
  }

  assert(false);
}

u64* strides_(param_t* p) {
  switch (p->type) {
    case TENSOR:
      return p->data.t->strides;
    case VIEW:
      return (u64*)p->data.v->strides;
  }

  assert(false);
}

f32 get_(param_t* p, u64 idxs[MAX_TENSOR_DIMS]) {
  switch (p->type) {
    case TENSOR:
      return tensor_get(p->data.t, idxs);
    case VIEW:
      return view_get(p->data.v, idxs);
  }

  assert(false);
}

void set_(param_t* p, u64 idxs[MAX_TENSOR_DIMS], f32 val) {
  switch (p->type) {
    case TENSOR:
      tensor_set(p->data.t, idxs, val);
      return;
    case VIEW:
      view_set(p->data.v, idxs, val);
      return;
  }

  assert(false);
}

bool same_shape(param_t* a, param_t* b) {
  u64 rank = rank_(a);
  if (rank != rank_(b)) return false;

  u64* dims_a = dims_(a);
  u64* dims_b = dims_(b);
  for (u64 dim = 0; dim < rank; ++dim)
    if (dims_a[dim] != dims_b[dim]) return false;

  return true;
}

param_t* clone(param_t* p) {
  ALLOC_PTR(param_t, c);
  c->type = p->type;

  switch (p->type) {
    case TENSOR:
      c->data.t = clone_tensor(p->data.t);
      return c;
    case VIEW:
      c->data.v = clone_view(p->data.v);
      return c;
  }

  assert(false);
}

void print(param_t* p) {
  switch (p->type) {
    case TENSOR:
      print_tensor(*((tensor_t**)p));
      return;
    case VIEW:
      print_view(*((tensor_view_t**)p));
      return;
  }

  assert(false);
}

void print_mat(param_t* p) {
  switch (p->type) {
    case TENSOR:
      print_tensor_mat(*((tensor_t**)p));
      return;
    case VIEW:
      print_view_mat(*((tensor_view_t**)p));
      return;
  }

  assert(false);
}
