#pragma once

#include <stdlib.h>

#include "num_types.h"
#include "voodoo.h"

typedef struct data_buff {
  f32* data;
  u64 n_el;
  u64 rc;
} data_buff;

data_buff* make_dat_buff(u64 n_el) {
  size_t data_size = sizeof(f32) * n_el;

  data_buff* d = malloc(sizeof(data_buff) + data_size);

  f32* data = ((f32*)(((u64)d) + sizeof(data_buff)));
  for (u64 i = 0; i < n_el; ++i) data[i] = 1.0 * (i + 1);

  d->data = data;
  d->n_el = n_el;
  d->rc = 1;

  return d;
}

data_buff* inc_rc(data_buff* d) {
  d->rc++;
  return d;
}

void dec_rc(data_buff* d) {
  d->rc--;
  if (d->rc == 0) free(d);
}

data_buff* clone_data_buff(data_buff* d) {
  ALLOC_PTR(data_buff, c);

  u64 n_el = d->n_el;
  size_t data_size = sizeof(f32) * n_el;
  c->data = malloc(data_size);
  memcpy(c->data, d->data, data_size);

  c->n_el = n_el;
  c->rc = 1;

  return c;
}
