#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
#include "dgemm.h"

#define SIZE 1 << 9
#define FUNC 5
#define AVX_512_FUNC 2
#define ALL_FUNC FUNC + AVX_512_FUNC

void checkResult(double ** restrict c, const int loop, const int size) {
    double epsilon = 1.0e-8;
    bool match = 1;
    for (int i = 0; i < size * size; i++) {
        if (abs(c[0][i] - c[loop][i]) > epsilon) {
            match = 0;
            printf("results do not match!\n");
            printf("Correct %5.2f Result %5.2f at current %d\n\n", c[0][i], c[loop][i], i);
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
    fp[1] = dgemm_avx2;
    fp[2] = dgemm_avx2_unroll;
    fp[3] = dgemm_avx2_unroll_block;
    fp[4] = dgemm_avx2_unroll_block_omp;
    #if defined (__AVX512F__) || defined (__AVX512__)
    nFunc = ALL_FUNC;
    fp[5] = dgemm_avx512;
    fp[6] = dgemm_avx512_unroll;
    #endif

	int n = SIZE;
    if (argc > 1) {
        n = 1 << atoi(argv[1]);
    }
    printf("Initializing matrices with size %d...\n", n);

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

	for (int i = 0; i < n * n; i++) {
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

    // execute dgemm kernels
    for (int i = 0; i < nFunc; i++) {
        switch (i) {
            case 0:
                printf("executing dgemm...\n");
                break;
            case 1:
                printf("executing dgemm_avx2...\n");
                break;
            case 2:
                printf("executing dgemm_avx2_unroll...\n");
                break;
            case 3:
                printf("executing dgemm_avx2_unroll_block...\n");
                break;
            case 4:
                printf("executing dgemm_avx2_unroll_block_omp...\n");
                break;
            case 5:
                printf("executing dgemm_avx512...\n");
                break;
            case 6:
                printf("executing dgemm_avx512_unroll...\n");
                break;
            default:
                break;
        }
        double dtime = - omp_get_wtime();
        fp[i](a, b, c[i], n);
        dtime += omp_get_wtime();
        printf("done, elapsed time: %.2f sec, ", dtime);
        if (i != 0) checkResult(c, i, n);
        else printf("\n\n");
    }

	return 0;
}
