#ifndef EPCUDAKERNEL_H
#define EPCUDAKERNEL_H


#ifdef __cplusplus
extern "C" {
#endif

#include "epblas/eputil.h"

eparseError_t vsPowx(long n, float *a, float b);

eparseError_t vsCosSinMatrixFast(long nrow, long ncol, const float* __restrict__ x, float* __restrict__ y);
#ifdef FAST_COSSIN
	#define vsCosSinMatrix vsCosSinMatrixFast
#else
	eparseError_t vsCosSinMatrix(long nrow, long ncol, const float* __restrict__ x, float* __restrict__ y) ;
#endif

eparseError_t vsScale(long n, float *x, float scaler);

eparseError_t vsInitx(long n, float *a, float b);

eparseError_t setCudaArrayByIndex(long idx, float *a, float v);

eparseError_t updateCudaArrayByIndex(long idx, float *a, float change);

eparseError_t cuda_saxpy(long n, float change, const float* __restrict__ x, long x_idx, float* __restrict__ y, long y_idx);

#ifdef __cplusplus
}
#endif


#endif
