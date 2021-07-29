# DGEMM
Basic DGEMM (Double-precision GEneral Matrix Multiplication) kernels with a bunch of optimizations such as SIMD vectorization, loop unrolling, blocking and multi-threading.

## Prerequisites
* CBLAS (For details, see: http://www.netlib.org/blas/)

## Files
* `main`: A main function which executes all the kernels and show the performance of each.
* `dgemm...`: Performs DGEMM for <img src="https://latex.codecogs.com/svg.latex?n&space;\times&space;n" title="n \times n" /> matrices in column major order (for easier SIMD vectorization).
* `...avx2 (or avx512)...`: A SIMD vectorized version of the kernel using AVX2 (or AVX-512) instruction intrinsics.
* `...unroll...`: A loop-unrolled version of the kernel. Unrolling factor is statically defined in `dgemm.h`.
* `...block...` : A blocked (tiled) version of the kernel. Block size is statically defined in `dgemm.h`.
* `...omp...`: A multi-threaded version of the kernel using OpenMP directives.

## Usage
Use `make run` to build and run the kernels. To enable AVX-512 versions, build with `make avx512=1` and run.

You can also specify the size of <img src="https://latex.codecogs.com/svg.latex?n" title="n" /> with `make run size=`*(a power of 2)*.
