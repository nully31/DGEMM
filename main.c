#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
#include <cblas.h>
#include "dgemm.h"

#define SIZE 1 << 9
#define FUNC 6
#define AVX_512_FUNC 4
#define ALL_FUNC FUNC + AVX_512_FUNC

void checkResult(double *ref, double ** restrict c, const int loop, const int size) {
    double epsilon = 1.0e-8;
    bool match = 1;
    for (int i = 0; i < size * size; i++) {
        if (abs(ref[i] - c[loop][i]) > epsilon) {
            match = 0;
            printf("results do not match!\n");
            printf("Correct %5.2f Result %5.2f at current %d\n\n", ref[i], c[loop][i], i);
            break;
        }
    }

    if (match) printf("results match.\n\n");
}

int main(int argc, char *argv[]) {
    printf("%s starting...\n", argv[0]);

    int nFunc = FUNC;
    void (*fp[ALL_FUNC])(double * restrict, double * restrict, double * restrict, const int);
    fp[0] = dgemm;
    fp[1] = dgemm_block;
    fp[2] = dgemm_avx2;
    fp[3] = dgemm_avx2_unroll;
    fp[4] = dgemm_avx2_unroll_block;
    fp[5] = dgemm_avx2_unroll_block_omp;
    #if defined (__AVX512F__) || defined (__AVX512__)
    nFunc = ALL_FUNC;
    fp[6] = dgemm_avx512;
    fp[7] = dgemm_avx512_unroll;
    fp[8] = dgemm_avx512_unroll_block;
    fp[9] = dgemm_avx512_unroll_block_omp;
    #endif

	int n = SIZE;
    if (argc > 1) {
        n = 1 << atoi(argv[1]);
    }
    printf("Initializing matrices with size %d...\n\n", n);

    // allocate memory
	/*
	__attribute__((aligned(32))) double a[n*n];
	__attribute__((aligned(32))) double b[n*n];
	__attribute__((aligned(32))) double c[n*n];
	*/
	size_t nBytes = n * n * sizeof(double);
	double *a = _mm_malloc(nBytes, 64);
	double *b = _mm_malloc(nBytes, 64);
    double **c = _mm_malloc(ALL_FUNC * nBytes, 64);
    for (int i = 0; i < nFunc; i++) {
        c[i] = _mm_malloc(nBytes, 64);
    }
    double *blas = _mm_malloc(nBytes, 64);

	for (int i = 0; i < n * n; i++) {
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

    // execute dgemm from blas library
    printf("executing blas...\n");
    double dtime = - omp_get_wtime();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, a, n, b, n, 1.0, blas, n);
    dtime += omp_get_wtime();
    printf("done, elapsed time: %.3f sec\n\n", dtime);

    // execute dgemm kernels
    for (int i = 0; i < nFunc; i++) {
        switch (i) {
            case 0:
                printf("executing dgemm...\n");
                break;
            case 1:
                printf("executing dgemm_block...\n");
                break;
            case 2:
                printf("executing dgemm_avx2...\n");
                break;
            case 3:
                printf("executing dgemm_avx2_unroll...\n");
                break;
            case 4:
                printf("executing dgemm_avx2_unroll_block...\n");
                break;
            case 5:
                printf("executing dgemm_avx2_unroll_block_omp...\n");
                break;
            case 6:
                printf("executing dgemm_avx512...\n");
                break;
            case 7:
                printf("executing dgemm_avx512_unroll...\n");
                break;
            case 8:
                printf("executing dgemm_avx512_unroll_block...\n");
                break;
            case 9:
                printf("executing dgemm_avx512_unroll_block_omp...\n");
                break;
            default:
                break;
        }
        dtime = - omp_get_wtime();
        fp[i](a, b, c[i], n);
        dtime += omp_get_wtime();
        printf("done, elapsed time: %.3f sec, ", dtime);
        checkResult(blas, c, i, n);
    }

	return 0;
}
