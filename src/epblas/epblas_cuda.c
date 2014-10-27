#include "epblas/epblas.h"
#include "epblas/epcudautil.h"
#include "epblas/epcudakernel.h"

static cublasHandle_t handle = NULL;

#define version "CUDA Enabled Embedding Parser BLAS"

void init(){
	if (handle == NULL)
		CUDABLAS_CHECK_RETURN(cublasCreate(&handle))
}

eparseError_t matrixDatacpyAnyToAny(Matrix_t dest, long dest_offset,
        Matrix_t src, long src_offset, size_t bytes) {
        
    debug("%s will be copied from src:+ld to dest:+%ld ", humanreadable_size(bytes), src_offset, dest_offset);

    if (dest == NULL || src == NULL)
        return eparseNullPointer;
    else {
        if (dest->dev == memoryCPU) {
            if (src->dev == memoryCPU) {
                memcpy(dest->data + dest_offset, src->data + src_offset, bytes);
            } else {
                cudaMemcpy(dest->data + dest_offset, src->data + src_offset,
                        bytes, cudaMemcpyDeviceToHost);
            }
        } else {
            if (src->dev == memoryCPU) {
                cudaMemcpy(dest->data + dest_offset, src->data + src_offset,
                        bytes, cudaMemcpyHostToDevice);
            } else {
                cudaMemcpy(dest->data + dest_offset, src->data + src_offset,
                        bytes, cudaMemcpyDeviceToDevice);
            }
        }
    }

    return eparseSucess;

}

eparseError_t ensureMatrixCapacity(Matrix_t mptr, long nrequired) {
	init();	

    if (mptr == NULL) {
        return eparseNullPointer;
    } else {
        if (nrequired <= mptr->capacity) {
            return eparseSucess;
        } else {
            long newCapacity = (long) ((nrequired * 6. / 2) + 1);
            
            
            log_info(
                    "Growing <%s>@%s capacity \tfrom %ld:%ld (%s)",
                    (mptr->identifier), 
                    ((mptr->dev == memoryGPU) ? "GPU" : "CPU"), (mptr->capacity), (mptr->n),
                    (humanreadable_size(sizeof(float) * (mptr->capacity))));
            log_info(
                    "\t\t\t\t\tto %ld:%ld (%s)",
                    newCapacity,nrequired,
                    (humanreadable_size(sizeof(float) * newCapacity)));
            log_info(
                    "\t\t\t\t\tfor %ld x %ld matrix", 
                    (mptr->nrow), (mptr->ncol));
            
            if (mptr->data == NULL) {
                if (mptr->dev == memoryGPU) {
                    CUDA_CHECK_RETURN(
                            cudaMalloc((void** )&(mptr->data),
                                    sizeof(float) * newCapacity));
                } else {
                    mptr->data = (float*) malloc(sizeof(float) * newCapacity);
                    
                    check_mem(mptr->data);
                }

            } 
            
            else {
                if (mptr->dev == memoryGPU) {
                    float *newPtr = NULL;

                    size_t freemem, totalmem;
                    CUDA_CHECK_RETURN(cudaMemGetInfo(&freemem, &totalmem))

                    if (freemem > sizeof(float) * newCapacity * 1.2) {
                        CUDA_CHECK_RETURN(
                                cudaMalloc((void** )&(newPtr),
                                        sizeof(float) * newCapacity))
                        check( newCapacity > nrequired && newCapacity > mptr->n, "New Capacity(%ld) should be larger than current number of elements (%ld) and required (%ld)",   newCapacity ,mptr->n,nrequired);  
                        
                        //CUDA_CHECK_RETURN(   cudaDeviceSynchronize() )
                        
                        check(newPtr != mptr->data, "Target and Source memory addresses are the same");
                        
                        debug("%d allocated and %d will be copied into this area", newCapacity,mptr->n );
                        check( mptr->data != NULL, "Ooops!!!. NULL pointer for GPU memory");
                        CUDA_CHECK_RETURN( cudaMemcpy(newPtr, mptr->data,
                                        				mptr->n * sizeof(float),
                                        				cudaMemcpyDeviceToDevice) )
                                        
                                  
                        
                        
                        //CUDA_CHECK_RETURN(   cudaDeviceSynchronize() )
                        
                        CUDA_CHECK_RETURN(cudaFree(mptr->data))

                        mptr->data = newPtr;
                    } 
                    else {
                        log_info(
                                "Low-performance cudaRealloc will be used due to limited memory");

                        newPtr = (float*) malloc(sizeof(float) * newCapacity);
                        CUDA_CHECK_RETURN(
                                cudaMemcpy(newPtr, mptr->data,
                                        mptr->n * sizeof(float),
                                        cudaMemcpyDeviceToHost))

                        CUDA_CHECK_RETURN(cudaFree(mptr->data))
                        CUDA_CHECK_RETURN(
                                cudaMalloc((void** )&(mptr->data),
                                        sizeof(float) * newCapacity))
                        CUDA_CHECK_RETURN(
                                cudaMemcpy(mptr->data, newPtr,
                                        mptr->n * sizeof(float),
                                        cudaMemcpyHostToDevice))
                        free(newPtr);

                    }

                } else {

                    mptr->data = (float*) realloc(mptr->data,
                            sizeof(float) * newCapacity);

                }

            }

            if (mptr->data != NULL) {
                mptr->capacity = newCapacity;

                return eparseSucess;
            } else
                return eparseMemoryAllocationError;
        }
    }
    
    error:
    	exit(EXIT_FAILURE);
}

eparseError_t newMatrix(Matrix_t *mptr, memoryAllocationDevice_t device,
        const char *id, long nrow, long ncol) {
        
    if (*mptr == NULL) {
        *mptr = (Matrix_t) malloc(sizeof(struct Matrix_st));

        check_mem( *mptr);

        

        if (id != NULL) {
            ((*mptr)->identifier) = strdup(id);
        } else {
            ((*mptr)->identifier) = strdup("noname");
        }
        
        (*mptr)->data = NULL;
        (*mptr)->capacity = 0;
        (*mptr)->n = 0;
        
        (*mptr)->dev = device ;
    }
    else{
    	check((*mptr)->dev == device, "You can not change memory type from %d to %d for %s", (*mptr)->identifier, (*mptr)->dev,  device);
    }

    
    (*mptr)->ncol = ncol;
    (*mptr)->nrow = nrow;

    EPARSE_CHECK_RETURN(ensureMatrixCapacity(*mptr, ncol * nrow));
    
    (*mptr)->n =  ncol * nrow;

    return eparseSucess;
    
    error:
    	return eparseMemoryAllocationError;
    	
}

eparseError_t vstackMatrix(Matrix_t *m1, memoryAllocationDevice_t device,
        const char* id, Matrix_t m2, bool transposeM2, bool releaseM2) {

    if (m2 == NULL)
        return eparseNullPointer;
    else {
        long copy_bytes = sizeof(float) * m2->n;
        long offset = 0;

        if (*m1 == NULL) {
            offset = 0;

            if (transposeM2)
            	EPARSE_CHECK_RETURN(newMatrix(m1,  device, id, m2->ncol,  m2->nrow))
            else
            	EPARSE_CHECK_RETURN(newMatrix(m1,  device, id, m2->nrow,  m2->ncol))
            	
            matrixDatacpyAnyToAny(*m1, offset, m2, 0, copy_bytes);
            
            #ifndef NDEBUG
            if ((*m1)->dev != m2->dev){
            	log_info("vstackMatrix calls matrixDatacpyAnyToAny");
            	if ((*m1)->dev == memoryCPU )
	            	log_info( "DtoH of %ld bytes",copy_bytes);
	            else
	            	log_info( "HtoD of %ld bytes",copy_bytes);
            }
            #endif

        } else {
        	check((*m1)->dev == device, "You can not change memory type from %d to %d for %s", (*m1)->identifier, (*m1)->dev,  device);
        	
            offset = (*m1)->n;

            if (((*m1)->ncol == m2->ncol && !transposeM2)
                    || ((*m1)->ncol == m2->nrow && transposeM2)) {
                
                if ( transposeM2 )
                	(*m1)->nrow += m2->ncol;
                else
                	(*m1)->nrow += m2->nrow;
        		
                EPARSE_CHECK_RETURN(ensureMatrixCapacity((*m1), (*m1)->n + m2->n))
            } else {
                return eparseColumnNumberMissmatch;

            }
            
            matrixDatacpyAnyToAny(*m1, offset, m2, 0, copy_bytes);
            
            #ifndef NDEBUG
            if ((*m1)->dev != m2->dev){
            	log_info("vstackMatrix calls matrixDatacpyAnyToAny");
            	if ((*m1)->dev == memoryCPU )
	            	log_info( "DtoH of %ld bytes",copy_bytes);
	            else
	            	log_info( "HtoD of %ld bytes",copy_bytes);
            }
            #endif

        	(*m1)->n += m2->n;
        }

        

    }

    if (releaseM2)
        deleteMatrix(m2);

    return eparseSucess;
    error:
    	return eparseMemoryAllocationError;
}

eparseError_t __deleteMatrix(Matrix_t m) {
if (m != NULL){
    if (m->dev == memoryCPU)
        free(m->data);
    else
        cudaFree(m->data);

    free(m);

    return eparseSucess;
}
else{
	log_warn("Nothing to free");
    return eparseSucess;
}
}


eparseError_t newInitializedMatrix(Matrix_t *mptr,
        memoryAllocationDevice_t device, const char *id, long nrow, long ncol,
        matrixInitializer_t strategy, float *fix_value,
        void *stream) {

    EPARSE_CHECK_RETURN(newMatrix(mptr, device, id, nrow, ncol));

    if (strategy == matrixInitNone) {
        ;
        //debug("Remember that matrix allocated contains garbage");
    } else if (strategy == matrixInitFixed) {

        if ((*mptr)->dev == memoryGPU) {
        	/*
            long i;
            float *temp = (float*) malloc(sizeof(float) * nrow * ncol);

            for (i = 0; i < nrow * ncol; i++)
                temp[i] = *fix_value;

            CUDA_CHECK_RETURN(cudaMemcpy((*mptr)->data, temp, nrow * ncol * sizeof(float),
                    cudaMemcpyHostToDevice))

            free(temp);
            */
            
            EPARSE_CHECK_RETURN(vsInitx( (*mptr)->n, (*mptr)->data, *fix_value))
        } else {
            for (int i = 0; i < nrow * ncol; i++)
                ((*mptr)->data)[i] = *fix_value;
        }
    } else {
        if (strategy != matrixInitRandom)
            debug("Unknown initialization strategy. Failed back to random");

        if ((*mptr)->dev == memoryGPU){
            ;

            //curandGenerateNormal((curandGenerator_t)stream, (*mptr)->data, nrow * ncol, 0., 1.);
        }
        else {
            for (int i = 0; i < nrow * ncol; i++)
                ((*mptr)->data)[i] = 0.0;
        }
    }


    //debug("Allocation and initialization for %lux%lu matrix took %f sec.\n",
    //		nrow, ncol, elapsed);

    return eparseSucess;
}

eparseError_t cloneMatrix(Matrix_t *dst, memoryAllocationDevice_t device,
        const Matrix_t src, const char *new_id) {

    if (src == NULL) {
        return eparseNullPointer;
    } else {
        EPARSE_CHECK_RETURN(
                newInitializedMatrix(dst, device, (new_id == NULL) ? (src->identifier) : new_id, src->nrow, src->ncol, matrixInitNone, NULL, NULL))

        EPARSE_CHECK_RETURN(matrixDatacpyAnyToAny(*dst, 0, src, 0, src->n * sizeof(float)));
        
        #ifndef NDEBUG
        if ((*dst)->dev != src->dev){
            	if ((*dst)->dev == memoryCPU )
	            	log_info( "DtoH of %ld bytes", src->n * sizeof(float));
	            else
	            	log_info( "HtoD of %ld bytes", src->n * sizeof(float));
        }
        #endif
        

        return eparseSucess;
    }

}

eparseError_t prodMatrixVector(Matrix_t A, bool tA, Vector_t x, Vector_t y){
	init();

    if (A->nrow == 0)
        return eparseSucess;
    else if (!((A->ncol == x->nrow && !tA) || (A->nrow == x->nrow && tA)))
        return eparseColumnNumberMissmatch;
    else{

        float alpha = 1., beta = 0.;

        check(A->dev == memoryGPU && x->dev ==memoryGPU, "Matrix(A) or Vector(x) is not stored in device memory");


        if (!tA){
            if(A->nrow != y->nrow){
                log_err("A(%ldx%ld) x x(%ld) does not conform with y(%ld)", A->nrow,A->ncol, x->nrow,y->nrow);
                return eparseColumnNumberMissmatch;
            }


            CUDABLAS_CHECK_RETURN(cublasSgemv(handle, CUBLAS_OP_N,
                    A->nrow,A->ncol,&alpha,
                    A->data,
                    A->nrow,x->data,1,&beta,y->data, 1))



        }
        else{
            if(A->ncol != y->nrow){
                log_err("A(%ldx%ld)^T x x(%ld) does not conform with y(%ld)", A->nrow,A->ncol, x->nrow,y->nrow);
                return eparseColumnNumberMissmatch;
            }

            CUDABLAS_CHECK_RETURN(cublasSgemv(handle, CUBLAS_OP_T,
                    A->nrow,A->ncol,&alpha,
                    A->data,
                    A->nrow,x->data,1,&beta,y->data, 1))
        }

        return eparseSucess;

    }

    error:
        return eparseMemoryAllocationError;
}

eparseError_t prodMatrixMatrix(Matrix_t A, Matrix_t B, bool tB, Matrix_t C){
	init();

    if (A->nrow == 0)
        return eparseSucess;
    else if( !((A->ncol == B->nrow && !tB) || (A->ncol == B->ncol && tB)) )
        return eparseColumnNumberMissmatch;
    else{
        float alpha = 1., beta = 0.;

        check(A->dev == memoryGPU && B->dev ==memoryGPU, "Matrix(A) or Matrix(B) is not stored in device memory");

        if (!tB){
            if( !(A->nrow == C->nrow && B->ncol == C->ncol) ){

                log_err("A(%ldx%ld) x B(%ldx%ld) does not conform with C(%ldx%ld)", A->nrow,A->ncol, B->nrow, B->ncol,C->nrow,C->ncol);
                return eparseColumnNumberMissmatch;
            }


            CUDABLAS_CHECK_RETURN(
            cublasSgemm( handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    A->nrow, B->ncol, A->ncol,
                    &alpha,
                    A->data, A->nrow,B->data, B->nrow,
                    &beta,
                    C->data, C->ncol))



        }
        else{
            if( !(A->nrow == C->nrow && B->nrow == C->ncol)){
                log_err( "A(%ldx%ld) x B(%ldx%ld)^T does not conform with C(%ldx%ld)", A->nrow,A->ncol, B->nrow, B->ncol,C->nrow,C->ncol);
                return eparseColumnNumberMissmatch;
            }


            CUDABLAS_CHECK_RETURN(
                    cublasSgemm( handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            A->nrow, B->nrow, A->ncol,
                            &alpha,
                            A->data, A->nrow,B->data, B->nrow,
                            &beta,
                            C->data, C->nrow))
        }


        return eparseSucess;

    }

    error:
        return eparseMemoryAllocationError;

}

eparseError_t powerMatrix(Matrix_t x, int power, Matrix_t y){
	/*
    if( !(x->nrow == y->nrow && x->ncol == y->ncol)){
        log_err( "x(%ldx%ld) and y(%ldx%ld) does not conform", x->nrow,x->ncol, y->nrow, y->ncol);
        return eparseColumnNumberMissmatch;
    }
    */

    check(x->dev == memoryGPU, "Vector(x) should be on GPU memory");
    
    EPARSE_CHECK_RETURN( vsPowx(x->n,x->data, power) )

    /*
    EPARSE_CHECK_RETURN(cloneMatrix(&x_host, memoryCPU, x, "x host"))
    
    for(int i =0;i<x_host->n;i++)
        (x_host->data)[i] = powf((x_host->data)[i],power);

	//    CUDABLAS_CHECK_RETURN(cublasSetVector(x_host->n, sizeof(float), x_host->data, 1, y->data, 1))
    
    
    

	EPARSE_CHECK_RETURN(matrixDatacpyAnyToAny(y, 0,
        x_host, 0, sizeof(float) * y->n))
        */

    //deleteMatrix(x_host)

    return eparseSucess;

    error:
        return eparseMemoryAllocationError;

}

eparseError_t dot(Vector_t x, Vector_t y, float *result){
    init();


    if( !(x->nrow == y->nrow && x->ncol == 1 &&  y->ncol == 1)){
        log_err( "x(%ldx%ld) and y(%ldx%ld) does not conform", x->nrow,x->ncol, y->nrow, y->ncol);

        return eparseColumnNumberMissmatch;
    }

    check(x->dev == memoryGPU && x->dev ==memoryGPU, "Vector(x) or Vector(y) is not stored in device memory");

    CUDABLAS_CHECK_RETURN(cublasSdot(handle, x->nrow, x->data, 1, y->data, 1, result))

    return eparseSucess;

    error:
        return eparseMemoryAllocationError;

}
