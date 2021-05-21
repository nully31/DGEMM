#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

void dgemm(int n, double *A, double *B, double *C) {
	for (int i = 0; i < n; i+=4) {
		for (int j = 0; j < n; j++) {
			__m256d c0 = _mm256_load_pd(C+i+j*n); /* cij = C[i][j] */
			for (int k = 0; k < n; k++)
				c0 = _mm256_add_pd(c0, /* cij = A[i][k] * B[k][j] */
							_mm256_mul_pd(_mm256_load_pd(A+i+k*n),
								_mm256_broadcast_sd(B+k+j*n)));
			_mm256_store_pd(C+i+j*n, c0); /* C[i][j] = cij */
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
