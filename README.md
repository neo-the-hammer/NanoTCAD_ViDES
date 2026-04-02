# NanoTCAD ViDES — Python 3.13 / NumPy 2.x / GCC 14 Compatible Fork

**NanoTCAD ViDES** is a self-consistent NEGF (Non-Equilibrium Green's Function) + Poisson solver for nanoscale device simulation, developed by Prof. Gianluca Fiori and Prof. Giuseppe Iannaccone at the University of Pisa.

This fork patches the original source code to compile and run on modern toolchains:

- **Python 3.13** (was: Python 2.7)
- **NumPy 2.x** (was: NumPy 1.x with deprecated C API)
- **GCC 14** (was: older GCC with relaxed defaults)

## What Was Patched

### Python 2 → 3 C API

| Old API (Python 2) | Replacement (Python 3) | Method |
|---|---|---|
| `PyInt_AsLong()` | `PyLong_AsLong()` | Preprocessor macro |
| `PyInt_FromLong()` | `PyLong_FromLong()` | Preprocessor macro |
| `PyString_AsString()` | `PyBytes_AsString(PyUnicode_AsEncodedString(...))` | Preprocessor macro + `#if PY_MAJOR_VERSION` guards |
| `Py_InitModule()` | `PyModule_Create()` with `PyModuleDef` | Conditional compilation in `NanoTCAD_ViDESmod.c` |

### NumPy 2.x C API

| Old API | Replacement | Method |
|---|---|---|
| `PyArray_DOUBLE` | `NPY_DOUBLE` | Preprocessor macro |
| `PyArray_CDOUBLE` | `NPY_CDOUBLE` | Preprocessor macro |
| `PyArray_INT` | `NPY_INT` | Preprocessor macro |
| `PyArray_FromDimsAndData()` | `PyArray_SimpleNewFromData()` | Compatibility shim in `compat_array.h` |
| NumPy headers at `numpy/core/include/` | `numpy/_core/include/` | Updated path in `configure.py` |

### GCC 14 Fortran/C Compatibility

| Flag | Purpose |
|---|---|
| `-std=gnu89` | Allow legacy C89 constructs |
| `-Wno-implicit-function-declaration` | Suppress missing prototype warnings |
| `-Wno-int-conversion` | Allow implicit int/pointer conversions |
| `-Wno-incompatible-pointer-types` | Allow mismatched pointer assignments |
| `-fallow-argument-mismatch` | Fortran: relax strict argument checking |
| `-std=legacy` | Fortran: allow F77 syntax |

### New Files Added

- `src/vides_compat.h` — Master compatibility header
- `src/compat_array.h` — `PyArray_FromDimsAndData()` replacement for NumPy 2.x

## Building

```bash
cd src
python3 configure.py
make
make install  # installs to lib/
```

The compiled shared library `NanoTCAD_ViDESmod.so` and Python wrapper `NanoTCAD_ViDES.py` are placed in `lib/`.

## Usage

```python
import sys
sys.path.insert(0, 'lib')
from NanoTCAD_ViDES import *

# Create a (10,0) zigzag CNT, 15 nm long
CNT = nanotube(10, 15)
print(f"Bandgap: {CNT.gap():.4f} eV")  # 0.9795 eV
```

See `demo/` for tutorial scripts (Id-Vgs sweeps, Poisson solutions, Schottky contacts, etc.).

## Credits

- **Original NanoTCAD ViDES**: Gianluca Fiori & Giuseppe Iannaccone, University of Pisa
  - Website: http://vides.nanotcad.com/vides/
  - Paper: Fiori & Iannaccone, *J. Comput. Electron.* (2023)
- **Python 3.13 / NumPy 2.x / GCC 14 patches**: Manish Jagdish Thatte

## License

Original NanoTCAD ViDES license applies. See `license.txt`.
