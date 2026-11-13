#!/usr/bin/bash

set -euxo pipefail

gcc --std=gnu23 -Wall -Wextra -pedantic -Wno-variadic-macros -lm -I ../.. xor.c -o out/xor
./xor
