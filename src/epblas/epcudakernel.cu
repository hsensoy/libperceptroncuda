#include "epblas/epcudakernel.h"


__global__ void _g_vsPowx(long n, float *a, float b) {
    for (long i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        a[i] = powf(a[i], b);
    }
}

__global__ void _g_vsScale(long n, float b, float *a) {
    for (long i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        a[i] = a[i]/b;
    }
}



__global__ void _g_vsCos(long n, float *a, float *b) {
    for (long i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        b[i] = cosf(a[i]);
    }
}

__global__ void _g_vsSin(long n, float *a, float *b) {
    for (long i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        b[i] = sinf(a[i]);
    }
}



__global__ void _g_vsCosSinFast(long nrow, long ncol, const float* __restrict__ a, float* __restrict__ b) {
    for (int ci = 0; ci < nrow * ncol; ci += nrow) {
        for (long i = blockIdx.x * blockDim.x + threadIdx.x;
             i < nrow;
             i += blockDim.x * gridDim.x) {
            sincosf(a[i + ci],&(b[2 * ci + i + nrow]),&(b[2 * ci + i]));
        }
    }
}

eparseError_t vsCosSinMatrixFast(long nrow, long ncol, const float* __restrict__ x, float* __restrict__ y) {

    _g_vsCosSinFast <<< 4096, 256 >>> (nrow, ncol, x, y);
    
    return eparseSucess;
}

__global__ void _g_vsCosSin(long n, const float* __restrict__ a, float* __restrict__ b) {
    for (long i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        sincosf(a[i],&(b[i+n]),&(b[i]));
    }
}

eparseError_t vsCosSinMatrix(long nrow, long ncol, const float* __restrict__ x, float* __restrict__ y) {

    /**
        todo: This loop can simply be removed and fully vectorized.
    */
    for (int i = 0; i < nrow * ncol; i += nrow) {
         _g_vsCosSin <<< 4096, 256 >>> (nrow, x + i, y + 2 * i);
    }

    return eparseSucess;
}


eparseError_t vsScale(long n,float *x,float scaler) {
    _g_vsScale <<< 4096, 256 >>> (n, scaler,x);

    return eparseSucess;
}

eparseError_t vsPowx(long n, float *a, float b) {


    _g_vsPowx <<< 4096, 256 >>> (n, a, b);


    return eparseSucess;
}


__global__ void _g_saxpy(long n, float change, const float* __restrict__ a, float* __restrict__ b) {
    for (long i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {

        b[i] += change * a[i];

    }
}

eparseError_t cuda_saxpy(long n, float change, const float* __restrict__ x, long x_idx, float* __restrict__ y, long y_idx){

    _g_saxpy<<< 4096, 256 >>> (n,change,x,y);
    
    return eparseSucess;
}


__global__ void _g_vsInitx(long n, float *a, float b) {
    for (long i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        a[i] = b;
    }

}


eparseError_t vsInitx(long n, float *a, float b) {
    _g_vsInitx <<< 4096, 256 >>> (n, a, b);

    return eparseSucess;

}

__global__ void _g_setArrayByIndex(long idx, float *a, float v) {

    a[idx] = v;

}


eparseError_t setCudaArrayByIndex(long idx, float *a, float v) {
    _g_setArrayByIndex <<< 1, 1 >>> (idx, a, v);

    return eparseSucess;

}


__global__ void _g_updateArrayByIndex(long idx, float *a, float change) {

    a[idx] += change;

}


eparseError_t updateCudaArrayByIndex(long idx, float *a, float change) {
    _g_updateArrayByIndex <<< 1, 1 >>> (idx, a, change);

    return eparseSucess;
}





