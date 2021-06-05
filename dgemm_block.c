#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#define SIZE 1 << 10
#define BLOCKSIZE 32

static void do_block(double * restrict A, double * restrict B, double * restrict C, const int n,
			const int si, const int sj, const int sk) {
	for (int i = si; i < si + BLOCKSIZE; i++) {
		for (int j = sk; j < sk + BLOCKSIZE; j++) {
			double cij = C[i+j*n]; /* cij = C[i][j] */
			for (int k = sj; k < sj + BLOCKSIZE; k++)
				cij += A[i+k*n] * B[k+j*n]; /* cij = A[i][k] * B[k][j] */
			C[i+j*n] = cij; /* C[i][j] = cij */
		}
	}
}

void dgemm_block(double * restrict A, double * restrict B, double * restrict C, const int n) {
	for (int si = 0; si < n; si += BLOCKSIZE) {
		for (int sj = 0; sj < n; sj += BLOCKSIZE) {
			for (int sk = 0; sk < n; sk += BLOCKSIZE) {
				do_block(A, B, C, n, si, sj, sk);
			}
		}
	}
}
