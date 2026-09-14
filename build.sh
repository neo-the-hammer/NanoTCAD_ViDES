#!/bin/sh
# Portable build for NanoTCAD ViDES.
#
# install.sh is a tcsh script, which many systems -- Google Colab among
# them -- do not have.  This does the same job in POSIX sh and adds the
# opt-in GPU build.
#
#   ./build.sh              CPU only (identical to what install.sh built)
#   ./build.sh --gpu        also compile the CUDA NEGF backend
#   ./build.sh --gpu --install
#
# --install copies the module into the Python prefix; without it the build
# products are left in src/ for you to use in place or copy yourself.
#
# Override the CUDA target if nvcc rejects the default architecture list:
#   CUDA_ARCH="-arch=sm_75" ./build.sh --gpu

set -e

GPU=0
INSTALL=0
for arg in "$@"; do
  case "$arg" in
    --gpu)     GPU=1 ;;
    --install) INSTALL=1 ;;
    -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

PYTHON=${PYTHON:-python3}
cd "$(dirname "$0")/src"

echo "Configuring for $($PYTHON --version 2>&1)"
$PYTHON ./configure.py

if [ "$GPU" = "1" ]; then
  if ! command -v "${NVCC:-nvcc}" >/dev/null 2>&1; then
    echo "error: nvcc not found. Install the CUDA toolkit, or set NVCC," >&2
    echo "       or drop --gpu to build the CPU-only module." >&2
    exit 1
  fi
  echo "Compiling NanoTCAD ViDES with the CUDA NEGF backend"
  make GPU=1
else
  echo "Compiling NanoTCAD ViDES (CPU only)"
  make
fi

if [ "$INSTALL" = "1" ]; then
  make ${GPU:+GPU=$GPU} install
  echo "Installed into $($PYTHON -c 'import sys; print(sys.prefix)')/lib/python*"
else
  echo
  echo "Built src/NanoTCAD_ViDESmod.so"
  echo "Use it with:  PYTHONPATH=$(pwd) python3 your_script.py"
fi
