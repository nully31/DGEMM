#include "dgemm.h"

void dgemm_avx2(double * restrict A, double * restrict B, double * restrict C, const int n) {
	for (int i = 0; i < n; i += 4) {
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