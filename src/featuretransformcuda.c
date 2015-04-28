//
// Created by husnu sensoy on 11/04/15.
//

#include "featuretransform.h"
#include <cuda.h>
#include <curand.h>
#include "epblas/epcudakernel.h"

static FeatureTransformer_t __newFeatureTransform(enum FeatureTransform type) {
    FeatureTransformer_t ft = (FeatureTransformer_t) malloc(sizeof(struct FeatureTransformer_st));

    check_mem(ft);

    ft->type = type;



    return ft;

    error:
    exit(EXIT_FAILURE);
}

RBFSampler_t __newRBFSampler(long D, float sigma) {

    RBFSampler_t  rbf = (RBFSampler_t)malloc(sizeof(struct RBFSampler_st));

    check_mem(rbf);

    rbf->d = -1;
    rbf->nsample = D;
    rbf->samples = NULL;
    rbf->sigma  =sigma;
    rbf->scaler = sqrtf(1./D);


    rbf->partial_inst = NULL;
    newInitializedGPUVector(&(rbf->partial_inst), "transformed partial",D,matrixInitNone,NULL,NULL)

    rbf->partial_matrix = NULL;
    rbf->in_cache = NULL;
    rbf->out_dev = NULL;

    return rbf;


    error:

    exit(EXIT_FAILURE);
}

static eparseError_t __initRBFSampler(RBFSampler_t pSt, long d) {
    if (pSt->d == -1){

        pSt->d = d;

        newInitializedGPUMatrix(&(pSt->samples),"fourier samples", pSt->nsample,d,matrixInitNone,NULL,NULL);

        curandGenerator_t gen;
        curandCreateGenerator(&gen,
                              CURAND_RNG_PSEUDO_DEFAULT);

        curandSetPseudoRandomGeneratorSeed(gen,
                                           777);

        log_info("Generating %ld gaussian vectors of %ld dimension with %f sigma",pSt->nsample , d ,pSt->sigma);


        curandStatus_t status = curandGenerateNormal(gen,pSt->samples->data, pSt->nsample * d, 0.0, pSt->sigma);

        if (status != CURAND_STATUS_SUCCESS){
            log_info("Error in CUDA random number generation %d", status);

            return eparseCUDAError;
        }

        curandDestroyGenerator(gen);
    }

    return eparseSucess;
}

FeatureTransformer_t newRBFSampler(long D, float sigma){

    FeatureTransformer_t ft = __newFeatureTransform(KERNAPROX_RBF_SAMPLER);

    ft->pDeriveObj = (void*) __newRBFSampler(D,sigma);

    return ft;
}

eparseError_t deleteFeatureTransformer(FeatureTransformer_t ft){
    // todo Implement this.

    return eparseSucess;
}

eparseError_t transform(FeatureTransformer_t ft, Vector_t in, Vector_t *out){
    RBFSampler_t rbf = NULL;
    float zero = 0.f;


    switch(ft->type){
        case KERNAPROX_RBF_SAMPLER:

            rbf = (RBFSampler_t)ft->pDeriveObj;
    	    newInitializedGPUVector(&(rbf->partial_inst), "transformed partial", rbf->nsample,matrixInitFixed,&zero,NULL)

            EPARSE_CHECK_RETURN(__initRBFSampler(rbf, in->n)    )

            if ( in->dev == memoryGPU) {
                EPARSE_CHECK_RETURN(prodMatrixVector(rbf->samples, false, in, rbf->partial_inst));
            }
            else{
                Vector_t in_dev = NULL;

                EPARSE_CHECK_RETURN(cloneVector(&in_dev, memoryGPU, in,"input on device"))

                EPARSE_CHECK_RETURN(prodMatrixVector(rbf->samples,false,in_dev , rbf->partial_inst));

                deleteVector(in_dev);
            }

            if( (*out) != NULL && (*out)->dev == memoryGPU ) {
                newInitializedGPUVector(out, "transformed", 2 * rbf->nsample, matrixInitNone, NULL, NULL)

                EPARSE_CHECK_RETURN(CosSinMatrix(rbf->partial_inst, *out));

                EPARSE_CHECK_RETURN(vsScale((*out)->n ,(*out)->data,rbf->scaler))

            }else{
                Vector_t out_dev = NULL;

                newInitializedGPUVector(&out_dev, "transformed", 2 * rbf->nsample, matrixInitNone, NULL, NULL)

                EPARSE_CHECK_RETURN(CosSinMatrix(rbf->partial_inst, out_dev))

                EPARSE_CHECK_RETURN(vsScale(out_dev->n ,out_dev->data, rbf->scaler))

                newInitializedCPUVector(out, "transformed", 2 * rbf->nsample, matrixInitNone, NULL, NULL)

                EPARSE_CHECK_RETURN(matrixDatacpyAnyToAny(*out,0,out_dev,0,out_dev->n * sizeof(float)))

                deleteVector(out_dev);

            }

            break;
        case KERNAPROX_NONE:

            cloneVector(out, memoryCPU,in, "transformed input");

            break;

        default:
            return eparseNotImplementedYet;

    }


    return eparseSucess;


}

eparseError_t __transformBatchOnDevice(FeatureTransformer_t ft, Matrix_t in, Matrix_t *out){
    RBFSampler_t rbf = NULL;
    float zero = 0.f;
    
    debug("__transformBatchOnDevice is called");
    
    check(   (*out) == NULL ||  (*out)->dev == memoryGPU, "__transformBatchOnDevice requires out Matrix to be on GPU memory");
    
    check( in->ncol <= TRANSFORM_BATCH_SIZE, "__transformBatchOnDevice is limited with %d instances at a time", TRANSFORM_BATCH_SIZE);
    
    
    switch(ft->type){
        case KERNAPROX_RBF_SAMPLER:

            rbf = (RBFSampler_t)ft->pDeriveObj;

            EPARSE_CHECK_RETURN(__initRBFSampler(rbf, in->nrow)    )
            
            EPARSE_CHECK_RETURN(newInitializedMatrix(out, memoryGPU, "Transformed Matrix on GPU", 2 * rbf->nsample, in->ncol, matrixInitNone,NULL,NULL))
            
            if (in->dev == memoryGPU)
                rbf->in_cache = in->dev ;
            else
                EPARSE_CHECK_RETURN(mtrxcolcpy(&( rbf->in_cache ), memoryGPU, in, "in GPU batch", 0, in->ncol))
                
            EPARSE_CHECK_RETURN(newInitializedMatrix(&(rbf->partial_matrix), memoryGPU, "partial matrix",rbf->nsample, in->ncol, matrixInitFixed, &zero, NULL))
               
            EPARSE_CHECK_RETURN(prodMatrixMatrix(rbf->samples,false,rbf->in_cache , rbf->partial_matrix))
                
            EPARSE_CHECK_RETURN(CosSinMatrix(rbf->partial_matrix, *out))

            EPARSE_CHECK_RETURN(vsScale((*out)->n ,(*out)->data, rbf->scaler))
                
            break;
        case KERNAPROX_NONE:

            cloneMatrix(out, memoryCPU,in, "transformed input");

            break;

        default:
            return eparseNotImplementedYet;

    }


    return eparseSucess;
    
error:
    return eparseCUDAError;

}


eparseError_t transformBatch(FeatureTransformer_t ft, Matrix_t in, Matrix_t *out){
    RBFSampler_t rbf = NULL;
    float zero = 0.f;
    
    debug("transformBatch is called");
        
    if ( (*out) != NULL && (*out)->dev == memoryGPU ) {
        return __transformBatchOnDevice(ft, in, out);
    }else{
        switch(ft->type){
            case KERNAPROX_RBF_SAMPLER:

                rbf = (RBFSampler_t)ft->pDeriveObj;

                EPARSE_CHECK_RETURN(__initRBFSampler(rbf, in->nrow)    )
                
                EPARSE_CHECK_RETURN(newInitializedMatrix(out, memoryCPU, "Transformed Matrix on CPU", 2 * rbf->nsample, in->ncol, matrixInitNone,NULL,NULL))
                
                long nleft = in->ncol;
                long offset = 0;
                
                while (nleft > 0) {
                    EPARSE_CHECK_RETURN(mtrxcolcpy(&( rbf->in_cache ), memoryGPU, in, "in GPU batch", offset, MIN(nleft, TRANSFORM_BATCH_SIZE)))
                    
                    EPARSE_CHECK_RETURN(newInitializedMatrix(&(rbf->partial_matrix), memoryGPU, "partial matrix",rbf->nsample, MIN(nleft, TRANSFORM_BATCH_SIZE), matrixInitFixed, &zero, NULL))
                    
                    EPARSE_CHECK_RETURN(prodMatrixMatrix(rbf->samples,false,rbf->in_cache , rbf->partial_matrix))
                    

                    EPARSE_CHECK_RETURN(newInitializedMatrix(&(rbf->out_dev),memoryGPU,"Transformed Matrix on Device",2 * rbf->nsample, MIN(nleft, TRANSFORM_BATCH_SIZE), matrixInitNone,NULL,NULL))

                    EPARSE_CHECK_RETURN(CosSinMatrix(rbf->partial_matrix, rbf->out_dev))

                    EPARSE_CHECK_RETURN(vsScale(rbf->out_dev->n ,rbf->out_dev->data, rbf->scaler))
                    
                    
                    EPARSE_CHECK_RETURN(matrixDatacpyAnyToAny(*out, offset * 2 * rbf->nsample, rbf->out_dev, 0,  2 * rbf->nsample * MIN(nleft, TRANSFORM_BATCH_SIZE) * sizeof(float)))
                    
                    offset +=  MIN(nleft, TRANSFORM_BATCH_SIZE);
                    nleft -= MIN(nleft, TRANSFORM_BATCH_SIZE);
                }
                    
                break;
            case KERNAPROX_NONE:

                cloneMatrix(out, memoryCPU,in, "transformed input");

                break;

            default:
                return eparseNotImplementedYet;

        }
        
        return eparseSucess;
    }
    
error:
    return eparseCUDAError;

}
