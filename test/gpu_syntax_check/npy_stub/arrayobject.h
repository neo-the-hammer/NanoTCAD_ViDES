/* Minimal stand-in for numpy/arrayobject.h.
 *
 * Like the CUDA stubs beside it, this exists only so the C sources can be
 * type-checked on a machine without NumPy's headers installed (see
 * check_syntax.sh).  It declares just the handful of NumPy entry points
 * ViDES actually uses.  NOT usable for a real build.
 */
#ifndef VIDES_FAKE_ARRAYOBJECT_H
#define VIDES_FAKE_ARRAYOBJECT_H

#include "Python.h"

typedef Py_intptr_t npy_intp;

typedef struct {
  PyObject_HEAD
  char    *data;
  int      nd;
  npy_intp *dimensions;
  npy_intp *strides;
} PyArrayObject;

enum {
  NPY_BOOL = 0, NPY_INT = 5, NPY_LONG = 7,
  NPY_FLOAT = 11, NPY_DOUBLE = 12, NPY_CFLOAT = 14, NPY_CDOUBLE = 15
};

PyObject *PyArray_SimpleNewFromData(int nd, npy_intp *dims, int typenum,
                                    void *data);
PyObject *PyArray_FromDims(int nd, int *dims, int typenum);
PyObject *PyArray_Return(PyArrayObject *ap);

#define import_array() do { } while (0)

#endif
