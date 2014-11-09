#include "epblas/epcudakernel.h"

#define MAX_GRIDSIZE 65535

__inline__ eparseError_t choose_dim(long n, int *grid_size, int *block_size, long *nactual)	{			
	if( n <= 1024 * MAX_GRIDSIZE ){
			if(n <= 128 * MAX_GRIDSIZE){	
				*grid_size = n / 128;		
				*block_size = 128;
				*nactual = n;
			}										
			else if( n <= 256 * MAX_GRIDSIZE ){		
				*grid_size = n / 256;				
				*block_size = 256;	
				*nactual = n;					
			}											
			else if( n <= 512 * MAX_GRIDSIZE ){	
				*grid_size = n / 512;				
				*block_size = 512;
				*nactual = n;						
			}										
			else{									
				*grid_size = n / 1024;					
				*block_size = 1024;	
				*nactual = n;						
			}
			
			
			if(n % (*block_size)) 
				++(*grid_size);			

			if ((*grid_size) > MAX_GRIDSIZE){
				log_err( "%d grid size with %d block_size", *grid_size,*block_size);		
				return eparseTooLargeCudaOp;
			}
			
			return eparseSucess;
		
	}	
	else{												
		log_err( "%ld total elements %d grid size with %d block_size", n, MAX_GRIDSIZE, 1024);		
		return eparseTooLargeCudaOp;
	}
}												

__global__ void _g_vsPowx(long n, float *a, float b) {	
	for (long i = blockIdx.x * blockDim.x + threadIdx.x; 
	         i < n; 
	         i += blockDim.x * gridDim.x) 
	      {
	          a[i] = powf(a[i], b);
	      }
}



eparseError_t vsPowx(long n, float *a, float b){	

	_g_vsPowx<<< 4096, 256 >>>(n, a, b);
    	
    return eparseSucess;

}


__global__ void _g_vsInitx(long n, float *a ,float b) {
	for (long i = blockIdx.x * blockDim.x + threadIdx.x; 
	         i < n; 
	         i += blockDim.x * gridDim.x) 
	      {
	          a[i] = b;
	      }

}



eparseError_t vsInitx(long n, float *a, float b){
	_g_vsInitx<<< 4096, 256 >>>(n, a, b);
    	
    return eparseSucess;

}

__global__ void _g_setArrayByIndex(long idx, float *a, float v) {
	
	a[idx] = v;

}


eparseError_t setCudaArrayByIndex(long idx, float *a, float v){
	 _g_setArrayByIndex<<< 1, 1 >>>(idx,a, v);
	 
	 return eparseSucess;

}


__global__ void _g_updateArrayByIndex(long idx, float *a, float change) {
	
	a[idx] += change;

}


eparseError_t updateCudaArrayByIndex(long idx, float *a, float change){
	_g_updateArrayByIndex<<< 1, 1 >>>(idx,a, change);
	
	return eparseSucess;
}





