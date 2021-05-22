#include "dgemm.h"

void dgemm_avx512_unroll(double * restrict A, double * restrict B, double * restrict C, const int n) {
	for (int i = 0; i < n; i += UNROLL_512 * 8) {
		for (int j = 0; j < n; j++) {
			__m512d c[UNROLL_512];
			for (int x = 0; x < UNROLL_512; x++)
				c[x] = _mm512_load_pd(C+i+x*8+j*n);	/* cij = C[i][j] */

			for (int k = 0; k < n; k++) {
				__m128d t0 = _mm_loadu_pd(B+k+j*n);
				__m512d b = _mm512_broadcastsd_pd(t0);
				for (int x = 0; x < UNROLL_512; x++)
					c[x] = _mm512_add_pd(c[x], 			/* cij = A[i][k] * B[k][j] */
								_mm512_mul_pd(_mm512_load_pd(A+i+k*n+x*8), b));
			}
			for (int x = 0; x < UNROLL_512; x++)
				_mm512_store_pd(C+i+x*8+j*n, c[x]); 		/* C[i][j] = cij */
		}
	}
}