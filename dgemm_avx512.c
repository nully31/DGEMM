#include "dgemm.h"

void dgemm_avx512(double * restrict A, double * restrict B, double * restrict C, const int n) {
	for (int i = 0; i < n; i+=8) {
		for (int j = 0; j < n; j++) {
			__m512d c0 = _mm512_load_pd(C+i+j*n); /* cij = C[i][j] */
			for (int k = 0; k < n; k++) {
				__m128d t0 = _mm_loadu_pd(B+k+j*n);
				c0 = _mm512_add_pd(c0, /* cij = A[i][k] * B[k][j] */
							_mm512_mul_pd(_mm512_load_pd(A+i+k*n),
								_mm512_broadcastsd_pd(t0)));
			}
			_mm512_store_pd(C+i+j*n, c0); /* C[i][j] = cij */
		}
	}
}