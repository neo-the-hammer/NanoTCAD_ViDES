// ======================================================================
//  Reference (CPU) implementation of the energy-batched RGF driver, and
//  the dispatcher that chooses between it and the CUDA backend.
//
//  The reference path deliberately does NOT reimplement the recursion.
//  It calls LDOS() / LDOS_Lake() / LDOSMODE() once per energy, exactly as
//  the scalar loop in CNT_charge_T.c and friends does today.  Two things
//  follow from that:
//
//    - Running an entry point through this driver with the GPU disabled
//      reproduces the stock numbers bit for bit, so the restructuring of
//      the energy loop can be validated on its own, without a GPU.
//    - It is an exact oracle for the CUDA backend: any disagreement
//      beyond round-off is a bug in the GPU path.
//
//  This file is released under the BSD license, as the rest of ViDES.
//  See "license.txt".
// ======================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vides_rgf_batch.h"
#include "nrutil.h"
#include "LDOS.h"
#include "LDOS_Lake.h"
#include "LDOSmode.h"

int vides_rgf_batch_cpu(const vides_rgf_desc *desc,
                        const double *E,
                        vides_complex ***diag,
                        vides_complex ***updiag,
                        vides_complex ***lowdiag,
                        vides_complex ***sigmas,
                        vides_complex ***sigmad,
                        double *A1, double *A2, double *T)
{
  const int n    = desc->n;
  const int Nc   = desc->Nc;
  const int NB   = desc->NB;
  const int nout = (desc->variant == VIDES_RGF_MODE) ? desc->Nreal : n;
  int b, i, j;

  if (n <= 0 || Nc <= 0 || NB <= 0) return 1;

  for (b = 0; b < NB; b++) {
    double **a1 = NULL, **a2 = NULL;
    double  t   = 0.0;

    switch (desc->variant) {
      case VIDES_RGF_LAKE:
        LDOS_Lake(E[b], lowdiag, diag, updiag, &a1, &a2,
                  sigmas[b], sigmad[b],
                  n, Nc, desc->flagtrans, &t, 0.0, desc->eta);
        break;
      case VIDES_RGF_MODE:
        LDOSMODE(E[b], lowdiag, diag, updiag, &a1, &a2,
                 sigmas[b], sigmad[b],
                 n, Nc, desc->flagtrans, &t,
                 desc->Nreal, (int *)desc->order, 0.0, desc->eta);
        break;
      case VIDES_RGF_STD:
      default:
        LDOS(E[b], lowdiag, diag, updiag, &a1, &a2,
             sigmas[b], sigmad[b],
             n, Nc, desc->flagtrans, &t, 0.0, desc->eta);
        break;
    }

    if (!a1 || !a2) return 1;

    for (i = 0; i < Nc; i++)
      for (j = 0; j < nout; j++) {
        A1[((size_t)b * Nc + i) * nout + j] = a1[i][j];
        A2[((size_t)b * Nc + i) * nout + j] = a2[i][j];
      }
    if (desc->flagtrans && T) T[b] = t;

    free_dmatrix(a1, 0, Nc - 1, 0, nout - 1);
    free_dmatrix(a2, 0, Nc - 1, 0, nout - 1);
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/* Dispatcher                                                          */
/* ------------------------------------------------------------------ */

int vides_rgf_batch(const vides_rgf_desc *desc,
                    const double *E,
                    vides_complex ***diag,
                    vides_complex ***updiag,
                    vides_complex ***lowdiag,
                    vides_complex ***sigmas,
                    vides_complex ***sigmad,
                    double *A1, double *A2, double *T)
{
  if (vides_gpu_available()) {
    int rc = vides_rgf_batch_gpu(desc, E, diag, updiag, lowdiag,
                                 sigmas, sigmad, A1, A2, T);
    if (rc == 0) return 0;
    /* A GPU failure is recoverable -- redo the batch on the CPU rather
       than losing the bias point.  Warn so it is not silent. */
    fprintf(stderr,
            "[ViDES/GPU] batch failed (rc=%d), falling back to CPU\n", rc);
  }
  return vides_rgf_batch_cpu(desc, E, diag, updiag, lowdiag,
                             sigmas, sigmad, A1, A2, T);
}

/* How many energies to handle per call.  On the GPU this is set by device
   memory; on the CPU a small chunk just amortises the per-call bookkeeping
   while keeping the temporary self-energy storage bounded. */
int vides_negf_chunk(const vides_rgf_desc *desc)
{
  const size_t nout = (size_t)((desc->variant == VIDES_RGF_MODE)
                               ? desc->Nreal : desc->n);
  size_t host_cap;
  int nb = 16;

  if (vides_gpu_available()) {
    int g = vides_gpu_batch_size(desc);
    if (g >= 1) nb = g;
  }

  /* The caller holds A1 and A2 for the whole chunk, so cap the batch so
     that pair stays under ~256 MB of host memory however much room the
     device reports. */
  host_cap = ((size_t)256 << 20)
             / (2 * (size_t)desc->Nc * (nout ? nout : 1) * sizeof(double));
  if (host_cap < 1) host_cap = 1;
  if ((size_t)nb > host_cap) nb = (int)host_cap;

  return nb < 1 ? 1 : nb;
}
