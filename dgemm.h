#ifndef _DGEMM_H_
#define _DGEMM_H_

#include <immintrin.h>

#define UNROLL 4
#define UNROLL_512 UNROLL * 2
#define BLOCKSIZE 32

void dgemm(double * restrict, double * restrict, double * restrict, const int);
void dgemm_avx2(double * restrict, double * restrict, double * restrict, const int);
void dgemm_avx2_unroll(double * restrict, double * restrict, double * restrict, const int);
void dgemm_avx2_unroll_block(double * restrict, double * restrict, double * restrict, const int);
void dgemm_avx2_unroll_block_omp(double * restrict, double * restrict, double * restrict, const int);
void dgemm_avx512(double * restrict, double * restrict, double * restrict, const int);
void dgemm_avx512_unroll(double * restrict, double * restrict, double * restrict, const int);

#endif