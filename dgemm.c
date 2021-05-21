#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

__attribute__((optimize("no-tree-vectorize")))
void dgemm(int n, double *A, double *B, double *C) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			double cij = C[i+j*n]; /* cij = C[i][j] */
			for (int k = 0; k < n; k++)
				cij += A[i+k*n] * B[k+j*n]; /* cij = A[i][k] * B[k][j] */
			C[i+j*n] = cij; /* C[i][j] = cij */
		}
	}
}

int main(int argc, char *argv[]) {
	int n = 3096;
	size_t size = n * n * sizeof(double);
	double *a = malloc(size);
	double *b = malloc(size);
	double *c = malloc(size);

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
