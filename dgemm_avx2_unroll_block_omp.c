#include "dgemm.h"

static void do_block(double * restrict A, double * restrict B, double * restrict C, const int n,
				const int si, const int sj, const int sk) {
	for (int i = si; i < si + BLOCKSIZE; i += UNROLL * 4)
		for (int j = sj; j < sj + BLOCKSIZE; j++) {
			__m256d c[UNROLL];
			for (int x = 0; x < UNROLL; x++)
				c[x] = _mm256_load_pd(C+i+x*4+j*n); 		/* c[x] = C[i][j] */

			for (int k = sk; k < sk + BLOCKSIZE; k++) {
				__m256d b = _mm256_broadcast_sd(B+k+j*n); 	/* b = B[k][j] */
				for (int x = 0; x < UNROLL; x++)
					c[x] = _mm256_add_pd(c[x],				/* c[x] += A[i][k] * b */
						_mm256_mul_pd(_mm256_load_pd(A+n*k+x*4+i), b));
			}
			for (int x = 0; x < UNROLL; x++)
				_mm256_store_pd(C+i+x*4+j*n, c[x]);			/* C[i][j] = c[x] */
	}
}

void dgemm_avx2_unroll_block_omp(double * restrict A, double * restrict B, double * restrict C, const int n) {
	#pragma omp parallel for
	for (int si = 0; si < n; si += BLOCKSIZE) {
		for (int sj = 0; sj < n; sj += BLOCKSIZE) {
			for (int sk = 0; sk < n; sk += BLOCKSIZE) {
				do_block(A, B, C, n, si, sj, sk);
			}
		}
	}
}