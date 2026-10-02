#!/usr/bin/bash

set -euxo pipefail

gcc -Wall -Wextra -pedantic num_types.h tensor.h tensor.c main.c -o shapetracker

./shapetracker
