#ifndef COMPAT_ARRAY_H
#define COMPAT_ARRAY_H

#include "arrayobject.h"

/* NumPy 2.x removed PyArray_FromDimsAndData.
   Provide a replacement using PyArray_SimpleNewFromData. */
#ifndef PyArray_FromDimsAndData
static PyObject *
PyArray_FromDimsAndData(int nd, int *dims, int typenum, char *data)
{
    npy_intp npy_dims[32];
    int i;
    for (i = 0; i < nd && i < 32; i++)
        npy_dims[i] = (npy_intp)dims[i];
    return PyArray_SimpleNewFromData(nd, npy_dims, typenum, (void *)data);
}
#endif

#endif /* COMPAT_ARRAY_H */
