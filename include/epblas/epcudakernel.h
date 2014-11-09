#ifndef EPCUDAKERNEL_H
#define EPCUDAKERNEL_H



#ifdef __cplusplus
extern "C" {
#endif

#include "epblas/eputil.h"
eparseError_t vsPowx(long n, float *a, float b);

eparseError_t vsInitx(long n, float *a, float b);

eparseError_t setCudaArrayByIndex(long idx, float *a, float v);
eparseError_t updateCudaArrayByIndex(long idx, float *a, float change);

#ifdef __cplusplus
}
#endif



#endif