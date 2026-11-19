#pragma once

#include <stdlib.h>
#include <string.h>

#include "num_types.h"
#include "voodoo.h"

typedef struct data_buff_t {
  f32* data;
  u64 n_el;
  u64 rc;
} data_buff_t;

#define ALLOC_DATA_BUFF(name, data_size, n_el)                 \
  data_buff_t* name = malloc(sizeof(data_buff_t) + data_size); \
  name->data = ((f32*)(((size_t)name) + sizeof(data_buff_t))); \
  name->n_el = n_el;                                           \
  name->rc = 1

data_buff_t* make_dat_buff(u64 n_el) {
  ALLOC_DATA_BUFF(d, (sizeof(f32) * n_el), n_el);

  for (u64 i = 0; i < n_el; ++i) d->data[i] = 1.0 * (i + 1);

  return d;
}

data_buff_t* inc_rc(data_buff_t* d) {
  d->rc++;
  return d;
}

void dec_rc(data_buff_t* d) {
  d->rc--;
  if (d->rc == 0) free(d);
}

data_buff_t* clone_data_buff(const data_buff_t* d) {
  u64 n_el = d->n_el;
  const size_t data_size = (sizeof(f32) * n_el);

  ALLOC_DATA_BUFF(c, data_size, n_el);
  memcpy(c->data, d->data, data_size);

  return c;
}
