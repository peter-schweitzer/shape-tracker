#include <inttypes.h>
#include <stdio.h>

#include "num_types.h"

static u32 xorshift32_state = 2463534242u;

f32 get_rand_f32() {
  xorshift32_state ^= xorshift32_state << 13;
  xorshift32_state ^= xorshift32_state >> 17;
  xorshift32_state ^= xorshift32_state << 5;

  f32 r = (f32)(((f64)xorshift32_state) / UINT32_MAX);
  printf("r: %f\n", r);
  return r;
}

void init_rand() {
  FILE* fd = fopen("/dev/urandom", "r");
  if (fd == NULL) {
    printf("couldn't open /dev/urandom, keeping current state " PRIu32, xorshift32_state);
    return;
  }

  fread(&xorshift32_state, sizeof(xorshift32_state), 1, fd);
  fclose(fd);
}
