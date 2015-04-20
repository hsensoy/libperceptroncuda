#ifndef EPCUDAKERNEL_H
#define EPCUDAKERNEL_H


#ifdef __cplusplus
extern "C" {
#endif

#include "epblas/eputil.h"

eparseError_t vsPowx(long n, float *a, float b);

eparseError_t vsCosSinMatrix(long nrow, long ncol, float *x, float *y);

eparseError_t vsScale(long n, float *x, float scaler);

eparseError_t vsInitx(long n, float *a, float b);

eparseError_t setCudaArrayByIndex(long idx, float *a, float v);

eparseError_t updateCudaArrayByIndex(long idx, float *a, float change);

eparseError_t cuda_saxpy(long n, float change, float *x, long x_idx, float *y, long y_idx);

#ifdef __cplusplus
}
#endif


#endif