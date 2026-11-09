#!/usr/bin/bash

set -euo pipefail

RED="\033[1;31m"
GREEN="\033[1;32m"
RESET="\033[0m"

C_ARGS=("--std=gnu23" "-Wall" "-Wextra" "-pedantic" "-Wno-variadic-macros" "-lm" "-I ..")

cd tests
tests=(*.c)

echo "found ${#tests[@]} test(s):"
for test in "${tests[@]}"; do
  echo "  $test"
done

echo ""

for test in "${tests[@]}"; do
  name="${test%.c}"
  printf "runing \"$name\" "

  if gcc ${C_ARGS[@]} $test -o out/$name && ./out/$name; then
    printf "${GREEN}passed${RESET}\n"
  else
    printf "${RED}failed${RESET}\n"
    exit 1
  fi
done
