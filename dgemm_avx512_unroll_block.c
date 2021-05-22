#include "dgemm.h"

void do_block_512(double * restrict A, double * restrict B, double * restrict C, const int n,
					const int si, const int sj, const int sk) {
	for (int i = si; i < si + BLOCKSIZE; i += UNROLL * 8) {
		for (int j = sj; j < sj + BLOCKSIZE; j++) {
			__m512d c[UNROLL];
			for (int x = 0; x < UNROLL; x++)
				c[x] = _mm512_load_pd(C+i+x*8+j*n);	/* cij = C[i][j] */

			for (int k = sk; k < sk + BLOCKSIZE; k++) {
				__m128d t0 = _mm_loadu_pd(B+k+j*n);
				__m512d b = _mm512_broadcastsd_pd(t0);
				for (int x = 0; x < UNROLL; x++)
					c[x] = _mm512_add_pd(c[x], 			/* cij = A[i][k] * B[k][j] */
								_mm512_mul_pd(_mm512_load_pd(A+i+k*n+x*8), b));
			}
			for (int x = 0; x < UNROLL; x++)
				_mm512_store_pd(C+i+x*8+j*n, c[x]); 		/* C[i][j] = cij */
		}
	}
}

void dgemm_avx512_unroll_block(double * restrict A, double * restrict B, double * restrict C, const int n) {
	for (int si = 0; si < n; si += BLOCKSIZE) {
		for (int sj = 0; sj < n; sj += BLOCKSIZE) {
			for (int sk = 0; sk < n; sk += BLOCKSIZE) {
				do_block_512(A, B, C, n, si, sj, sk);
			}
		}
	}
}