#include "epblas/epcudakernel.h"



__global__ void _g_vsPowx(long n, float *a, float b) {
	int idx = threadIdx.x +  blockIdx.x * blockDim.x;

	if (idx < n)
		a[idx] = powf(a[idx], b);
}



eparseError_t vsPowx(long n, float *a, float b){

	if ( n <= 1000 )
    	_g_vsPowx<<< 32, 512 >>>(n,a,  (float)b);
    else
    	_g_vsPowx<<< 16, 1024 >>>(n,a, (float)b);
    	
    return eparseSucess;

}


__global__ void _g_vsInitx(long n, float *a ,float b) {
	int idx = threadIdx.x +  blockIdx.x * blockDim.x;

	if (idx < n)
		a[idx] = b;
}



eparseError_t vsInitx(long n, float *a, float b){

	if ( n <= 1000 )
    	_g_vsInitx<<< 32, 512 >>>(n,a,  b);
    else
    	_g_vsInitx<<< 16, 1024 >>>(n,a, b);
    	
    return eparseSucess;

}