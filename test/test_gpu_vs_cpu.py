#!/usr/bin/env python3
"""Check that the CUDA NEGF backend reproduces the CPU result.

Both paths run the same recursion; the GPU one just runs it for many
energies at once, so the two should agree to round-off.  They will not
agree bit for bit -- a batched cuBLAS gemm sums in a different order than
the reference zgemm -- so this compares against a tolerance and prints the
actual deviation it saw rather than only pass/fail.

The backend is selected through the VIDES_GPU environment variable, which
is read once per process, so each case is run twice in fresh subprocesses.

    python3 test/test_gpu_vs_cpu.py              # every case
    python3 test/test_gpu_vs_cpu.py --case gnr   # just one
    python3 test/test_gpu_vs_cpu.py --rtol 1e-9  # stricter

Run it from the directory holding NanoTCAD_ViDESmod.so, or point
PYTHONPATH at it.  On a machine with no GPU both runs land on the CPU and
the test is vacuous -- it says so rather than reporting a pass.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np


# ----------------------------------------------------------------------
# Cases.  Each builds a small device and returns (charge, E, T).
# Keep them small: this is a correctness check, not a benchmark.
# ----------------------------------------------------------------------

def case_cnt():
    """CNT real space -> CNT_charge_T -> VIDES_RGF_STD."""
    from NanoTCAD_ViDES import nanotube
    d = nanotube(10, 2.0)
    d.Elower, d.Eupper, d.dE = -3.0, 3.0, 0.02
    d.Phi = -0.2 * np.ones(d.n * d.Nc)
    d.mu2 = -0.3
    d.charge_T()
    return d.charge, d.E, d.T


def case_cntmode():
    """CNT mode space -> CNTmode_charge_T -> VIDES_RGF_MODE."""
    from NanoTCAD_ViDES import nanotube
    d = nanotube(10, 2.0)
    d.Nmodes = 4
    d.Elower, d.Eupper, d.dE = -3.0, 3.0, 0.02
    d.Phi = -0.2 * np.ones(d.n * d.Nc)
    d.mu2 = -0.3
    d.mode_charge_T()
    return d.charge, d.E, d.T


def case_gnr():
    """Graphene nanoribbon -> GNR_charge_T -> VIDES_RGF_STD."""
    from NanoTCAD_ViDES import nanoribbon
    d = nanoribbon(3, 1)
    d.Elower, d.Eupper, d.dE = -3.0, 3.0, 0.02
    d.Phi = -0.2 * d.z
    d.mu2 = -0.3
    d.charge_T()
    return d.charge, d.E, d.T


def case_hamiltonian():
    """Generic tight-binding Hamiltonian -> H_charge_T -> VIDES_RGF_LAKE.

    This is the same path the Zincblend / silicon-nanowire class takes, so
    covering it covers that too.
    """
    from NanoTCAD_ViDES import Hamiltonian
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))
    from GNR import GNR
    h = GNR(5, 6)
    d = Hamiltonian(5, 6)
    d.H = h
    d.Elower, d.Eupper, d.dE = -3.0, 3.0, 0.02
    d.eta = 1e-5
    d.charge_T()
    return d.charge, d.E, d.T


CASES = {
    "cnt": case_cnt,
    "cntmode": case_cntmode,
    "gnr": case_gnr,
    "hamiltonian": case_hamiltonian,
}


# ----------------------------------------------------------------------
# Worker: run one case under the backend the environment selects.
# ----------------------------------------------------------------------

def worker(name, outfile):
    charge, E, T = CASES[name]()
    n = min(len(E), len(T))
    np.savez(outfile,
             charge=np.asarray(charge, dtype=float),
             E=np.asarray(E, dtype=float)[:n],
             T=np.asarray(T, dtype=float)[:n],
             gpu=np.array([1 if os.environ.get("VIDES_GPU", "1") != "0" else 0]))


def run_backend(name, use_gpu, outfile):
    env = dict(os.environ)
    env["VIDES_GPU"] = "1" if use_gpu else "0"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [os.getcwd(), env.get("PYTHONPATH", "")] if p)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker", name, outfile],
        env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise RuntimeError("%s run of case '%s' failed"
                           % ("GPU" if use_gpu else "CPU", name))
    return proc.stdout


def deviation(a, b):
    """Max absolute difference and max relative difference."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return float("inf"), float("inf")
    d = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    nz = scale > 0
    rel = np.zeros_like(d)
    rel[nz] = d[nz] / scale[nz]
    return (float(d.max()) if d.size else 0.0,
            float(rel.max()) if rel.size else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2, metavar=("CASE", "OUT"),
                    help=argparse.SUPPRESS)
    ap.add_argument("--case", choices=sorted(CASES), action="append",
                    help="run only this case (repeatable)")
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="max tolerated relative deviation (default 1e-6)")
    args = ap.parse_args()

    if args.worker:
        worker(args.worker[0], args.worker[1])
        return 0

    names = args.case or sorted(CASES)
    tmp = tempfile.mkdtemp(prefix="vides_gpu_check_")
    failures, skipped = [], []

    for name in names:
        print("=" * 68)
        print("case: %s" % name)
        print("=" * 68)

        cpu_out = os.path.join(tmp, name + "_cpu.npz")
        gpu_out = os.path.join(tmp, name + "_gpu.npz")

        run_backend(name, False, cpu_out)
        banner = run_backend(name, True, gpu_out)

        for line in banner.splitlines():
            if "NEGF backend" in line:
                print("  " + line.strip())
                break

        cpu = np.load(cpu_out)
        gpu = np.load(gpu_out)

        if "no CUDA device" in banner or "GPU disabled" in banner:
            print("  no usable GPU -- both runs used the CPU, nothing compared")
            skipped.append(name)
            continue

        worst = 0.0
        for key in ("charge", "T", "E"):
            adiff, rdiff = deviation(cpu[key], gpu[key])
            worst = max(worst, rdiff)
            status = "ok" if rdiff <= args.rtol else "FAIL"
            print("  %-7s max|dGPU-CPU| = %-12.4g  max rel = %-12.4g  %s"
                  % (key, adiff, rdiff, status))

        if worst > args.rtol:
            failures.append(name)

    print("=" * 68)
    if skipped:
        print("SKIPPED (no GPU): %s" % ", ".join(skipped))
    if failures:
        print("FAILED: %s" % ", ".join(failures))
        return 1
    if not skipped:
        print("all cases agree within rtol=%g" % args.rtol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
