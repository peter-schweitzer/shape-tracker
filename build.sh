#!/usr/bin/bash

set -euxo pipefail

gcc --std=gnu23 -Wall -Wextra -pedantic main.c -o shapetracker

./shapetracker
