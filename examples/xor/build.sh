#!/usr/bin/bash

set -euxo pipefail

# note: asan is only intended for debugging the example
# gcc --std=gnu23 -Wall -Wextra -pedantic -Wno-variadic-macros -lm -I ../.. xor.c -o out/xor -fsanitize=address
gcc --std=gnu23 -Wall -Wextra -pedantic -Wno-variadic-macros -lm -I ../.. xor.c -o out/xor -Ofast -march=native

./out/xor
