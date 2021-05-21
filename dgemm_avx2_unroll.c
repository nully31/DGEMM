#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

#define UNROLL (4)

void dgemm(int n, double *A, double *B, double *C) {
	for (int i = 0; i < n; i+=UNROLL*4) {
		for (int j = 0; j < n; j++) {
			__m256d c[4];
			for (int x = 0; x < UNROLL; x++)
				c[x] = _mm256_load_pd(C+i+x*4+j*n);

			for (int k = 0; k < n; k++) {
				__m256d b = _mm256_broadcast_sd(B+k+j*n);
				for (int x = 0; x < UNROLL; x++)
					c[x] = _mm256_add_pd(c[x],
						_mm256_mul_pd(_mm256_load_pd(A+n*k+x*4+i), b));
			}
			for (int x = 0; x < UNROLL; x++)
				_mm256_store_pd(C+i+x*4+j*n, c[x]);
		}
	}
}

int main(int argc, char *argv[]) {
	int n = 3096;
	/*
	__attribute__((aligned(32))) double a[n*n];
	__attribute__((aligned(32))) double b[n*n];
	__attribute__((aligned(32))) double c[n*n];
	*/
	size_t size = n * n * sizeof(double);
	double *a = _mm_malloc(size, 32);
	double *b = _mm_malloc(size, 32);
	double *c = _mm_malloc(size, 32);

	for (int i = 0; i < n * n; i++) {
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

	double dtime = - omp_get_wtime();
	dgemm(n, a, b, c);
	dtime += omp_get_wtime();

	printf("C[N-1][N-1] = %lf\t", c[n*n-1]);
	printf("%lf sec\n", dtime);

	return 0;
}
