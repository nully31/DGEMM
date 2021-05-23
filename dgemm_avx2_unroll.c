#include "dgemm.h"

void dgemm_avx2_unroll(double * restrict A, double * restrict B, double * restrict C, const int n) {
	for (int i = 0; i < n; i += UNROLL * 4)
		for (int j = 0; j < n; j++) {
			__m256d c[UNROLL];
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