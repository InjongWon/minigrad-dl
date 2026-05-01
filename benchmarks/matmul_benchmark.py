"""
benchmarks/matmul_benchmark.py
================================
Compare matrix multiplication speed:
  1. NumPy (MKL/OpenBLAS)
  2. MiniGrad CPU (tiled C++ / Python fallback)
  3. MiniGrad CUDA (if available)

Run: python benchmarks/matmul_benchmark.py

This demonstrates that our hand-written C++ tiled matmul is competitive
(within 2-3× of NumPy) and that the CUDA version can outperform NumPy.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from minigrad.autograd import Tensor

SIZES = [256, 512, 1024, 2048]
REPS = 5


def bench_numpy(n):
    A = np.random.randn(n, n).astype(np.float32)
    B = np.random.randn(n, n).astype(np.float32)
    # warmup
    _ = A @ B
    t0 = time.perf_counter()
    for _ in range(REPS):
        _ = A @ B
    return (time.perf_counter() - t0) / REPS * 1000  # ms


def bench_minigrad_cpu(n):
    A = Tensor(np.random.randn(n, n).astype(np.float32))
    B = Tensor(np.random.randn(n, n).astype(np.float32))
    # warmup
    _ = (A @ B).numpy()
    t0 = time.perf_counter()
    for _ in range(REPS):
        _ = (A @ B).numpy()
    return (time.perf_counter() - t0) / REPS * 1000


def bench_minigrad_cuda(n):
    try:
        from minigrad.backend_ndarray.ndarray import cuda as cuda_dev
        import minigrad.backend_ndarray.ndarray as nd
        dev = cuda_dev()
        A = nd.NDArray(np.random.randn(n, n).astype(np.float32), device=dev)
        B = nd.NDArray(np.random.randn(n, n).astype(np.float32), device=dev)
        # warmup
        _ = (A @ B).numpy()
        t0 = time.perf_counter()
        for _ in range(REPS):
            _ = (A @ B).numpy()
        return (time.perf_counter() - t0) / REPS * 1000
    except Exception:
        return None


def main():
    print("=" * 65)
    print(f"  Matrix Multiply Benchmark  ({REPS} runs, avg ms)")
    print("=" * 65)
    print(f"{'Size':>8}  {'NumPy':>10}  {'MG-CPU':>10}  {'MG-CUDA':>10}  {'CPU/NP':>8}")
    print("-" * 65)

    for n in SIZES:
        np_ms = bench_numpy(n)
        cpu_ms = bench_minigrad_cpu(n)
        cuda_ms = bench_minigrad_cuda(n)
        ratio = cpu_ms / np_ms
        cuda_str = f"{cuda_ms:10.2f}" if cuda_ms is not None else "    N/A   "
        print(
            f"{n:8d}  {np_ms:10.2f}  {cpu_ms:10.2f}  {cuda_str}  {ratio:7.2f}×"
        )

    print("=" * 65)
    print(
        "\nNote: MG-CPU uses tiled C++ matmul (no BLAS).\n"
        "      NumPy uses MKL/OpenBLAS — so 2-3× slower is expected.\n"
        "      CUDA should outperform NumPy for large matrices.\n"
        "      N/A = CUDA not available / not built."
    )


if __name__ == "__main__":
    main()