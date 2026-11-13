#include <libc.h>
#include <stdio.h>

#include "num_types.h"

#define INIT_RAND_BUFF_SIZE 4

static u32 xorshift32_state = 2463534242u;

f32 get_rand_f32() {
  xorshift32_state ^= xorshift32_state << 13;
  xorshift32_state ^= xorshift32_state >> 17;
  xorshift32_state ^= xorshift32_state << 5;

  f32 r = (f32)(((f64)xorshift32_state) / UINT32_MAX);
  printf("r: %f\n", r);
  return r;
}

// no error handling because it's gonna work and if not there are other problems 🤷‍♂️
void init_rand() {
  u32 buff = xorshift32_state;  // keep using current state as fallback
  FILE* fd = fopen("/dev/urandom", "r");
  fread(&buff, sizeof(buff), 1, fd);
  fclose(fd);
  printf("old state: %u | new state: %u\n", xorshift32_state, buff);
  xorshift32_state = buff;
}
