#!/usr/bin/bash

set -euxo pipefail

gcc --std=gnu23 -Wall -Wextra -pedantic -Wno-variadic-macros -lm main.c -o shapetracker

./shapetracker
