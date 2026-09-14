// ======================================================================
//  CPU-only stubs for the GPU entry points.
//
//  This file is compiled instead of vides_gpu.cu when the tree is built
//  without GPU=1, so that the dispatcher in vides_rgf_batch.c links and
//  behaves sensibly (it simply never leaves the CPU path).
//
//  This file is released under the BSD license, as the rest of ViDES.
//  See "license.txt".
// ======================================================================
#include "vides_rgf_batch.h"

int vides_gpu_compiled(void) { return 0; }
int vides_gpu_available(void) { return 0; }
int vides_gpu_batch_size(const vides_rgf_desc *desc) { (void)desc; return 0; }

const char *vides_gpu_describe(void)
{
  return "CPU only (built without GPU=1)";
}

int vides_rgf_batch_gpu(const vides_rgf_desc *desc,
                        const double *E,
                        vides_complex ***diag,
                        vides_complex ***updiag,
                        vides_complex ***lowdiag,
                        vides_complex ***sigmas,
                        vides_complex ***sigmad,
                        double *A1, double *A2, double *T)
{
  (void)desc; (void)E; (void)diag; (void)updiag; (void)lowdiag;
  (void)sigmas; (void)sigmad; (void)A1; (void)A2; (void)T;
  return -1;   /* never selected: vides_gpu_available() is 0 */
}
