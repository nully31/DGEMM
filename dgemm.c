void dgemm(double * restrict A, double * restrict B, double * restrict C, const int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			double cij = C[i+j*n]; /* cij = C[i][j] */
			for (int k = 0; k < n; k++)
				cij += A[i+k*n] * B[k+j*n]; /* cij = A[i][k] * B[k][j] */
			C[i+j*n] = cij; /* C[i][j] = cij */
		}
	}
}