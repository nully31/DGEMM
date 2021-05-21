#ifndef _DGEMM_H_
#define _DGEMM_H_

#include <immintrin.h>

void dgemm(double * restrict, double * restrict, double * restrict, const int);
void dgemm_avx2(double * restrict, double * restrict, double * restrict, const int);
void dgemm_avx2_unroll(double * restrict, double * restrict, double * restrict, const int);
void dgemm_avx2_unroll_block(double * restrict, double * restrict, double * restrict, const int);
void dgemm_avx512(double * restrict, double * restrict, double * restrict, const int);

extern void (*fp[])(double * restrict, double * restrict, double * restrict, const int);

#endif