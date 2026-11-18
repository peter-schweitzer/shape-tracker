#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#include "num_types.h"

// NOTE: assuming default terminal size of 80 x 24
void plot(const u64 EPOCHS, const f32* losses) {
  f32 max = -INFINITY;
  f32 min = INFINITY;

  for (u64 i = 0; i < EPOCHS; ++i) {
    max = losses[i] > max ? losses[i] : max;
    min = losses[i] < min ? losses[i] : max;
  }

  const u64 PLOT_WIDTH = 71;

  f32 loss_range = (max - min);
  f32 loss_step = loss_range / 18.0f;
  f32 loss_label_step = loss_range / 10.0f;

  printf(
      "\n\
LOSS   A\n\
%5.3f ─┤\n\
       │\n\
%5.3f ─┤\n\
       │\n\
%5.3f ─┤\n\
       │\n\
%5.3f ─┤\n\
       │\n\
%5.3f ─┤\n\
       │\n\
%5.3f ─┤\n\
       │\n\
%5.3f ─┤\n\
       │\n\
%5.3f ─┤\n\
       │\n\
%5.3f ─┤\n\
       │\n\
%5.3f ─┤\n\
       └┬──────────────────────────────────┬──────────────────────────────────┬>\n\
        0                             %7" PRIu64 "                      EPOCH %7" PRIu64 "\n",
      max, max - loss_label_step, max - loss_label_step * 2, max - loss_label_step * 3, max - loss_label_step * 4, max - loss_label_step * 5, max - loss_label_step * 6, max - loss_label_step * 7, max - loss_label_step * 8, min, EPOCHS / 2, EPOCHS);

  const f32 epoch_step = (float)EPOCHS / (float)(PLOT_WIDTH + 1);
  for (u64 x = 0; x < PLOT_WIDTH; ++x) {
    printf("\033[s");
    printf("\r");
    f32 loss = losses[(u64)roundf(x * epoch_step)];
    u64 y = (u64)roundf((loss - min) / loss_step);
    printf("\033[%" PRIu64 "A", 3 + y);
    printf("\033[%" PRIu64 "C", 8 + x);
    printf("*");
    printf("\033[u");
  }
}
