#include "epblas/epblas.h"
#include "epblas/epcudautil.h"
#include "epblas/epcudakernel.h"

static cublasHandle_t handle = NULL;

#define VERSION "CUDA Enabled Embedding Parser BLAS " "0.0.7.1"

#ifdef NDEBUG
#define EPBLAS_PROMPT "\nLoading:" "\n" "epblas " VERSION " - " "Production" "\n\n"
#else
    #define EPBLAS_PROMPT "\nLoading:" "\n" "epblas " VERSION " - " "Debug" "\n\n"
#endif

#define GPU_MEMORY_GROWTH_RATE 1.2
#define CPU_MEMORY_GROWTH_RATE 1.5


void init() {
    if (handle == NULL) {
	log_info("%s", EPBLAS_PROMPT);
        CUDABLAS_CHECK_RETURN(cublasCreate(&handle))
    }
}

eparseError_t memcpyAnyToAny(float *dest, long dest_offset, memoryAllocationDevice_t dest_dev,
                             float *src, long src_offset, memoryAllocationDevice_t src_dev, size_t bytes) {

    if (dest == NULL || src == NULL)
        return eparseNullPointer;
    else {
        if (dest_dev == memoryCPU) {
            if (src_dev == memoryCPU) {
                memcpy(dest + dest_offset, src + src_offset, bytes);
            } else {
                cudaMemcpy(dest + dest_offset, src + src_offset,
                           bytes, cudaMemcpyDeviceToHost);
            }
        } else {
            if (src_dev == memoryCPU) {
                cudaMemcpy(dest + dest_offset, src + src_offset,
                           bytes, cudaMemcpyHostToDevice);
            } else {
                cudaMemcpy(dest + dest_offset, src + src_offset,
                           bytes, cudaMemcpyDeviceToDevice);
            }
        }
    }

    return eparseSucess;


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
        if (nrequired < mptr->capacity) {
            return eparseSucess;
        } else {
            long newCapacity;

            if (mptr->dev == memoryGPU)
                newCapacity = (long) ((nrequired * GPU_MEMORY_GROWTH_RATE) + 1);
            else
                newCapacity = (long) ((nrequired * CPU_MEMORY_GROWTH_RATE) + 1);


            debug(
                    "Growing <%s>@%s capacity \tfrom %ld:%ld (%s)",
                    (mptr->identifier),
                    ((mptr->dev == memoryGPU) ? "GPU" : "CPU"), (mptr->capacity), (mptr->n),
                    (humanreadable_size(sizeof(float) * (mptr->capacity))));
            debug(
                    "\t\t\t\t\tto %ld:%ld (%s)",
                    newCapacity, nrequired,
                    (humanreadable_size(sizeof(float) * newCapacity)));
            debug(
                    "\t\t\t\t\tfor %ld x %ld matrix",
                    (mptr->nrow), (mptr->ncol));

            if (mptr->data == NULL) {
                if (mptr->dev == memoryGPU) {
                    CUDA_CHECK_RETURN(
                            cudaMalloc((void **) &(mptr->data),
                                       sizeof(float) * newCapacity));
                } else {
                    mptr->data = (float *) malloc(sizeof(float) * newCapacity);

                    check(mptr->data != NULL, "Memory allocation error of %ld bytes", sizeof(float) * newCapacity);
                }

            }

            else {
                if (mptr->dev == memoryGPU) {
                    float *newPtr = NULL;

                    size_t freemem, totalmem;
                    CUDA_CHECK_RETURN(cudaMemGetInfo(&freemem, &totalmem))

                    if (freemem > sizeof(float) * newCapacity * GPU_MEMORY_GROWTH_RATE) {
                        CUDA_CHECK_RETURN(
                                cudaMalloc((void **) &(newPtr),
                                           sizeof(float) * newCapacity))
                        check(newCapacity > nrequired && newCapacity > mptr->n,
                              "New Capacity(%ld) should be larger than current number of elements (%ld) and required (%ld)",
                              newCapacity, mptr->n, nrequired);

                        //CUDA_CHECK_RETURN(   cudaDeviceSynchronize() )

                        check(newPtr != mptr->data, "Target and Source memory addresses are the same");

                        debug("%d allocated and %d will be copied into this area", newCapacity, mptr->n);
                        check(mptr->data != NULL, "Ooops!!!. NULL pointer for GPU memory");
                        CUDA_CHECK_RETURN(cudaMemcpy(newPtr, mptr->data,
                                                     mptr->n * sizeof(float),
                                                     cudaMemcpyDeviceToDevice))




                        //CUDA_CHECK_RETURN(   cudaDeviceSynchronize() )

                        CUDA_CHECK_RETURN(cudaFree(mptr->data))

                        mptr->data = newPtr;
                    }
                    else {
                        log_info(
                                "Low-performance cudaRealloc will be used due to limited memory");

                        newPtr = (float *) malloc(sizeof(float) * newCapacity);
                        CUDA_CHECK_RETURN(
                                cudaMemcpy(newPtr, mptr->data,
                                           mptr->n * sizeof(float),
                                           cudaMemcpyDeviceToHost))

                        CUDA_CHECK_RETURN(cudaFree(mptr->data))
                        CUDA_CHECK_RETURN(
                                cudaMalloc((void **) &(mptr->data),
                                           sizeof(float) * newCapacity))
                        CUDA_CHECK_RETURN(
                                cudaMemcpy(mptr->data, newPtr,
                                           mptr->n * sizeof(float),
                                           cudaMemcpyHostToDevice))
                        free(newPtr);

                    }

                } else {

                    mptr->data = (float *) realloc(mptr->data,
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

        check_mem(*mptr);


        if (id != NULL) {
            ((*mptr)->identifier) = strdup(id);
        } else {
            ((*mptr)->identifier) = strdup("noname");
        }

        (*mptr)->data = NULL;
        (*mptr)->capacity = 0;
        (*mptr)->n = 0;

        (*mptr)->dev = device;
    }
    else {
        check((*mptr)->dev == device, "You can not change memory type for %s from %d to %d", (*mptr)->identifier,
              (*mptr)->dev, device);
    }


    (*mptr)->ncol = ncol;
    (*mptr)->nrow = nrow;

    EPARSE_CHECK_RETURN(ensureMatrixCapacity(*mptr, ncol * nrow));

    (*mptr)->n = ncol * nrow;

    return eparseSucess;

    error:
    return eparseMemoryAllocationError;

}

eparseError_t vstackMatrix(Matrix_t *m1, memoryAllocationDevice_t device,
                           const char *id, Matrix_t m2, bool transposeM2, bool releaseM2) {

    if (m2 == NULL)
        return eparseNullPointer;
    else {
        long copy_bytes = sizeof(float) * m2->n;
        long offset = 0;

        if (*m1 == NULL) {
            offset = 0;

            if (transposeM2) EPARSE_CHECK_RETURN(newMatrix(m1, device, id, m2->ncol, m2->nrow))
            else EPARSE_CHECK_RETURN(newMatrix(m1, device, id, m2->nrow, m2->ncol))

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
            check((*m1)->dev == device, "You can not change memory type for %s from %d to %d", (*m1)->identifier,
                  (*m1)->dev, device);

            offset = (*m1)->n;

            if (((*m1)->ncol == m2->ncol && !transposeM2)
                || ((*m1)->ncol == m2->nrow && transposeM2)) {

                if (transposeM2)
                    (*m1)->nrow += m2->ncol;
                else
                    (*m1)->nrow += m2->nrow;

                EPARSE_CHECK_RETURN(ensureMatrixCapacity((*m1), (*m1)->n + m2->n))
            } else {
                log_err("Column mismatch ");
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

    if (releaseM2) deleteMatrix(m2);

    return eparseSucess;
    error:
    return eparseMemoryAllocationError;
}


eparseError_t vappend(Vector_t *v, memoryAllocationDevice_t device, const char *id, float value) {

    long copy_bytes = sizeof(float) * 1;
    long offset = 0;

    if (*v == NULL) {
        offset = 0;

        newVector(v, device, id, 1)

        float temp = value;

        if (device == memoryCPU) {
            memcpy((*v)->data, &temp, copy_bytes);
        } else {
            cudaMemcpy((*v)->data, &temp, copy_bytes, cudaMemcpyHostToDevice);
        }
    } else {
        check((*v)->dev == device, "You can not change memory type of %s from %d to %d", (*v)->identifier, (*v)->dev,
              device);

        offset = (*v)->n;


        (*v)->nrow += 1;

        EPARSE_CHECK_RETURN(ensureMatrixCapacity((*v), (*v)->n + 1))


        float temp = value;

        if (device == memoryCPU) {
            memcpy((*v)->data + offset, &temp, copy_bytes);
        } else {
            cudaMemcpy((*v)->data + offset, &temp, copy_bytes, cudaMemcpyHostToDevice);
        }

        (*v)->n += 1;
    }

    return eparseSucess;
    error:
    return eparseMemoryAllocationError;
}

eparseError_t vappend_array(Vector_t *v, memoryAllocationDevice_t device, const char *id, long n, float *arr) {

    check(n > 0, "Number of elements to be appended should be positive whereas %ld found", n);
    long copy_bytes = sizeof(float) * n;
    long offset = 0;

    if (*v == NULL) {
        offset = 0;

        newVector(v, device, id, n)

        EPARSE_CHECK_RETURN(memcpyAnyToAny((*v)->data, 0, device, arr, 0, memoryCPU, copy_bytes))

    } else {

        offset = (*v)->n;


        (*v)->nrow += n;

        EPARSE_CHECK_RETURN(ensureMatrixCapacity((*v), (*v)->n + n))


        EPARSE_CHECK_RETURN(memcpyAnyToAny((*v)->data, offset, (*v)->dev, arr, 0, memoryCPU, copy_bytes))

        (*v)->n += n;
    }

    return eparseSucess;
    error:
    return eparseMemoryAllocationError;


}

eparseError_t vappend_vector(Vector_t *v, memoryAllocationDevice_t device, const char *id, const Vector_t in) {

    check(in != NULL, "Input vector can not be null");


    long offset = 0;

    if (*v == NULL) {

        EPARSE_CHECK_RETURN(cloneVector(v, device, in, id))


    } else {

        offset = (*v)->n;

        EPARSE_CHECK_RETURN(ensureMatrixCapacity((*v), (*v)->n + in->n))
        EPARSE_CHECK_RETURN(matrixDatacpyAnyToAny(*v, offset, in, 0, sizeof(float) * in->n))

        (*v)->nrow += in->nrow;
        (*v)->n += in->n;
    }

    return eparseSucess;
    error:
    return eparseMemoryAllocationError;

}

bool vequal(const Vector_t v1, const Vector_t v2) {

    if (v1 == v2)
        return true;

    check(v1 != NULL && v2 != NULL, "One of the vectors is uninitialized.");
    check(v1->n == v2->n, "Number of elements v1(%ld) and v2(%ld) do not match", v1->n, v2->n);


    Vector_t v1_onCPU = NULL, v2_onCPU = NULL;
    if (v1->dev == memoryGPU) {

        EPARSE_CHECK_RETURN(cloneVector(&v1_onCPU, memoryCPU, v1, "Clone v1"))


    } else {
        v1_onCPU = v1;
    }

    if (v2->dev == memoryGPU) {

        EPARSE_CHECK_RETURN(cloneVector(&v2_onCPU, memoryCPU, v2, "Clone v2"))


    } else {
        v2_onCPU = v2;
    }


    for (long i = 0; i < v1_onCPU->n; ++i) {
        check((v1_onCPU->data)[i] == (v2_onCPU->data)[i],
              "Elements at position %ld does not match in v1(%f) and v2(%f)", i, (v1_onCPU->data)[i],
              (v2_onCPU->data)[i]);
    }

    if (v1->dev == memoryGPU) {
        deleteVector(v1_onCPU)
    }

    if (v2->dev == memoryGPU) {
        deleteVector(v2_onCPU)
    }

    return true;
    error:
    return false;
}

eparseError_t hstack(Matrix_t *m1, memoryAllocationDevice_t device, const char *id, Matrix_t m2, bool transposeM2,
                     bool releaseM2) {

    if (m2 == NULL)
        return eparseNullPointer;
    else {
        long copy_bytes = sizeof(float) * m2->n;
        long offset = -1000000;

        if (*m1 == NULL) {
            offset = 0;


            EPARSE_CHECK_RETURN(newMatrix(m1, device, id, 0, 0))

            EPARSE_CHECK_RETURN(ensureMatrixCapacity(*m1, m2->n))
        } else {
            offset = (*m1)->n;

            if (((*m1)->nrow == m2->nrow && !transposeM2)
                || ((*m1)->nrow == m2->ncol && transposeM2)) {
                EPARSE_CHECK_RETURN(ensureMatrixCapacity((*m1), (*m1)->n + m2->n))
            } else {
                log_err("%s(%ldx%ld) %s(%ldx%ld) row numbers mismatch", (*m1)->identifier, (*m1)->nrow, (*m1)->ncol,
                        m2->identifier, m2->nrow, m2->ncol);
                return eparseColumnNumberMissmatch;

            }
        }

        EPARSE_CHECK_RETURN(matrixDatacpyAnyToAny(*m1, offset,
                                                  m2, 0, copy_bytes))

        if (transposeM2) {
            (*m1)->nrow = m2->ncol;
            (*m1)->ncol += m2->nrow;
        } else {
            (*m1)->nrow = m2->nrow;
            (*m1)->ncol += m2->ncol;
        }

        (*m1)->n += m2->n;

    }

    if (releaseM2) deleteMatrix(m2);

    return eparseSucess;

}

eparseError_t __deleteMatrix(Matrix_t m) {
    if (m != NULL) {
        if (m->dev == memoryCPU)
            free(m->data);
        else
            cudaFree(m->data);

        free(m);

        return eparseSucess;
    }
    else {
        log_warn("Nothing to free");

        return eparseSucess;
    }
}


eparseError_t newInitializedMatrix(Matrix_t *mptr,
                                   memoryAllocationDevice_t device, const char *id, long nrow, long ncol,
                                   matrixInitializer_t strategy, float *fix_value,
                                   void *stream) {

    EPARSE_CHECK_RETURN(newMatrix(mptr, device, id, nrow, ncol));

    if (strategy == matrixInitNone) { ;
        //debug("Remember that matrix allocated contains garbage");
    } else if (strategy == matrixInitFixed) {

        if ((*mptr)->dev == memoryGPU) {
            /*
            Matrix_t temp = NULL;
            EPARSE_CHECK_RETURN(newMatrix(&temp, memoryCPU, "temp", nrow, ncol));
            for(size_t i = 0; i < temp->n; ++i)
                (temp->data)[i] = *fix_value;

            EPARSE_CHECK_RETURN(cloneMatrix(mptr, memoryGPU, temp, id))

            EPARSE_CHECK_RETURN(__deleteMatrix(temp))
            */

            EPARSE_CHECK_RETURN(vsInitx((*mptr)->n, (*mptr)->data, *fix_value))

        } else {
            for (int i = 0; i < nrow * ncol; i++)
                ((*mptr)->data)[i] = *fix_value;
        }
    } else if (strategy == matrixInitCArray) {

        if ((*mptr)->dev == memoryGPU) {
            log_warn("Unknown initialization strategy. Failed back to random");

            return eparseFailOthers;

        } else {
            for (int i = 0; i < nrow * ncol; i++)
                ((*mptr)->data)[i] = fix_value[i];
        }
    }
    else {
        if (strategy != matrixInitRandom)
            log_warn("Unknown initialization strategy. Failed back to random");

        //TODO: Implement random initialization for GPU and CPU Matrix
        if ((*mptr)->dev == memoryGPU) { ;

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
                newInitializedMatrix(dst, device, (new_id == NULL) ? (src->identifier) : new_id, src->nrow, src->ncol,
                                     matrixInitNone, NULL, NULL))

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

eparseError_t mtrxcolcpy(Matrix_t *dst, memoryAllocationDevice_t device,
                         const Matrix_t src, const char *new_id, long offsetcol, long ncol) {
    if (src == NULL) {
        return eparseNullPointer;
    } else {
        check(ncol > 0, "Number of columns to be copied (ncol) should be positive");
        check(offsetcol >= 0 && offsetcol < src->ncol,
              "Column offset (offsetcol) should be between [0, # of columns source)");

        EPARSE_CHECK_RETURN(
                newInitializedMatrix(dst, device, (new_id == NULL) ? (src->identifier) : new_id, src->nrow,
                                     MIN(src->ncol, ncol), matrixInitNone, NULL, NULL))

        EPARSE_CHECK_RETURN(matrixDatacpyAnyToAny(*dst, 0, src, offsetcol * src->nrow,
                                                  MIN(src->ncol, ncol) * src->nrow * sizeof(float)));

        return eparseSucess;
    }

    error:
    return eparseFailOthers;
}


eparseError_t prodMatrixVector(Matrix_t A, bool tA, Vector_t x, Vector_t y) {
    init();

    if (A->nrow == 0)
        return eparseSucess;
    else if (!((A->ncol == x->nrow && !tA) || (A->nrow == x->nrow && tA))) {
        log_err("A(%ldx%ld)[%s] x x(%ld) does not conform with y(%ld)", A->nrow, A->ncol, (tA ? "tranposed" : ""),
                x->nrow, y->nrow);
        return eparseColumnNumberMissmatch;
    }
    else {

        float alpha = 1.f, beta = 1.f;

        check(A->dev == memoryGPU && x->dev == memoryGPU, "Matrix(A) or Vector(x) is not stored in device memory");


        if (!tA) {
            if (A->nrow != y->nrow) {
                log_err("A(%ldx%ld) x x(%ld) does not conform with y(%ld)", A->nrow, A->ncol, x->nrow, y->nrow);
                return eparseColumnNumberMissmatch;
            }


            CUDABLAS_CHECK_RETURN(cublasSgemv(handle, CUBLAS_OP_N,
                                              A->nrow, A->ncol, &alpha,
                                              A->data,
                                              A->nrow, x->data, 1, &beta, y->data, 1))


        }
        else {
            if (A->ncol != y->nrow) {
                log_err("A(%ldx%ld)^T x x(%ld) does not conform with y(%ld)", A->nrow, A->ncol, x->nrow, y->nrow);
                return eparseColumnNumberMissmatch;
            }

            CUDABLAS_CHECK_RETURN(cublasSgemv(handle, CUBLAS_OP_T,
                                              A->nrow, A->ncol, &alpha,
                                              A->data,
                                              A->nrow, x->data, 1, &beta, y->data, 1))
        }

        return eparseSucess;

    }

    error:
    return eparseMemoryAllocationError;
}

eparseError_t prodMatrixMatrix(Matrix_t A, bool tA, Matrix_t B, Matrix_t C) {
    init();

    if (A->nrow == 0)
        return eparseSucess;
    else {
        float alpha = 1.f, beta = 1.f;

        check(A->dev == memoryGPU && B->dev == memoryGPU, "Matrix(A) or Matrix(B) is not stored in device memory");

        if (!tA) {
            if (!(A->ncol == B->nrow && A->nrow == C->nrow && B->ncol == C->ncol)) {

                log_err("A(%ldx%ld) x B(%ldx%ld) does not conform with C(%ldx%ld)", A->nrow, A->ncol, B->nrow, B->ncol,
                        C->nrow, C->ncol);
                return eparseColumnNumberMissmatch;
            }


            CUDABLAS_CHECK_RETURN(
                    cublasSgemm(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                A->nrow, B->ncol, A->ncol,
                                &alpha,
                                A->data, A->nrow, B->data, B->nrow,
                                &beta,
                                C->data, C->nrow))


        }
        else {
            if (!(A->nrow == B->nrow && A->ncol == C->nrow && B->ncol == C->ncol)) {
                log_err("A(%ldx%ld)^T x B(%ldx%ld) does not conform with C(%ldx%ld)", A->nrow, A->ncol, B->nrow,
                        B->ncol, C->nrow, C->ncol);
                return eparseColumnNumberMissmatch;
            }


            CUDABLAS_CHECK_RETURN(
                    cublasSgemm(handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                A->ncol, B->ncol, A->nrow,
                                &alpha,
                                A->data, A->nrow, B->data, B->nrow,
                                &beta,
                                C->data, C->nrow))
        }


        return eparseSucess;

    }

    error:
    return eparseMemoryAllocationError;

}

eparseError_t powerMatrix(Matrix_t x, int power, Matrix_t y) {
    /*
    if( !(x->nrow == y->nrow && x->ncol == y->ncol)){
        log_err( "x(%ldx%ld) and y(%ldx%ld) does not conform", x->nrow,x->ncol, y->nrow, y->ncol);
        return eparseColumnNumberMissmatch;
    }
    */

    check(x->dev == memoryGPU, "Vector(x) should be on GPU memory");
    check(y == NULL, "powerMatrix implementation on CUDA is inplace. Set y to NULL and check x for output.")

    EPARSE_CHECK_RETURN(vsPowx(x->n, x->data, power))


    return eparseSucess;

    error:
    return eparseMemoryAllocationError;

}

eparseError_t CosSinMatrix(Matrix_t x, Matrix_t y) {
    if (!(2 * x->nrow == y->nrow && x->ncol == y->ncol)) {
        log_err("x(%ldx%ld) and y(%ldx%ld) does not conform", x->nrow, x->ncol, y->nrow, y->ncol);
        return eparseColumnNumberMissmatch;
    }

    check(x->dev == memoryGPU, "Matrix(x) should be on GPU memory");
    check(y->dev == memoryGPU, "Matrix(y) should be on GPU memory");

    EPARSE_CHECK_RETURN(vsCosSinMatrix(x->nrow, x->ncol, x->data, y->data))


    return eparseSucess;

    error:
    return eparseMemoryAllocationError;

}

eparseError_t dot(Vector_t x, Vector_t y, float *result) {
    init();


    if (!(x->nrow == y->nrow && x->ncol == 1 && y->ncol == 1)) {
        log_err("x(%ldx%ld) and y(%ldx%ld) does not conform", x->nrow, x->ncol, y->nrow, y->ncol);

        return eparseColumnNumberMissmatch;
    }

    check(x->dev == memoryGPU && x->dev == memoryGPU, "Vector(x) or Vector(y) is not stored in device memory");

    CUDABLAS_CHECK_RETURN(cublasSdot(handle, x->nrow, x->data, 1, y->data, 1, result))

    return eparseSucess;

    error:
    return eparseMemoryAllocationError;

}

void printMatrixVerbose(const char *heading, Matrix_t m, FILE *fp, long max_row, long max_col) {
    //todo: Implement this for CUDA also.
}
