#!/bin/sh
# Type-check src/vides_gpu.cu on a machine with no CUDA toolkit.
#
# nvcc is the real check, but it needs a toolkit that a plain workstation
# (or a CI runner) may not have.  This script gets most of the way there:
# it strips the <<< >>> launch configurations, so kernel launches read as
# ordinary calls, and compiles the result as C++ against the stub headers
# next to this file.  That catches typos, wrong argument counts, wrong
# types and undeclared identifiers in both the kernels and the host code.
#
# What it does NOT check: anything nvcc alone knows about -- shared memory
# sizing, launch bounds, device-side codegen, and of course whether the
# numerics are right.  Run test/test_gpu_vs_cpu.py on a real GPU for that.
#
# Usage:  sh test/gpu_syntax_check/check_syntax.sh

set -e
here=$(cd "$(dirname "$0")" && pwd)
src=$here/../../src
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

perl -0pe 's/<<<.*?>>>//gs' "$src/vides_gpu.cu" > "$tmp/vides_gpu.cpp"

g++ -fsyntax-only -std=c++11 -x c++ \
    -I"$here" -I"$src" \
    -Wall -Wno-unused-function -Wno-unused-variable \
    "$tmp/vides_gpu.cpp"

echo "vides_gpu.cu: host-side type check passed"
