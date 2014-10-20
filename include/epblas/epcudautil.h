#ifndef EPCUDAUTIL_H
#define	EPCUDAUTIL_H

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "curand.h"

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

#define CUDABLAS_CHECK_RETURN(value) {											\
	cublasStatus_t _m_cudablasStat = value;										\
	if (_m_cudablasStat != CUBLAS_STATUS_SUCCESS) {										\
		fprintf(stderr, "Error %d at line %d in file %s\n",					\
				_m_cudablasStat, __LINE__, __FILE__);		\
		exit(1);															\
	} }

#endif