/* Compatibility shim for Python 3 + NumPy 2.x
   This file is included by NanoTCAD_ViDESmod.c after Python.h/arrayobject.h */
#ifndef VIDES_COMPAT_H
#define VIDES_COMPAT_H

/* Include the PyArray_FromDimsAndData wrapper */
#include "compat_array.h"

#endif /* VIDES_COMPAT_H */
