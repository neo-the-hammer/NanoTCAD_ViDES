#ifndef COMPLEX_H
#define COMPLEX_H

/* Workaround: C99/C11 <complex.h> defines 'complex' as a keyword.
   We need our own struct type. Use a typedef with a unique name
   and macro-alias it back to 'complex' for source compatibility. */
typedef struct{
  double r;
  double i;
} vides_complex;

/* Only define the macro if C99 complex hasn't already claimed it */
#ifdef complex
#undef complex
#endif
#define complex vides_complex

#endif
