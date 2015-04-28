#include "simpleperceptron.h"
#include "epcudakernel.h"


#define BATCH_SIZE 1200 //For simple perceptron we have more space to be used at CUDA hardware

eparseError_t deleteSimplePerceptron(SimplePerceptron_t sp) {
    deleteVector(sp->best_w);
    deleteVector(sp->w);
    deleteVector(sp->w_avg);
    deleteFeatureTransformer(sp->ft);

    free(sp);

    return eparseSucess;

}

SimplePerceptron_t __newSimplePerceptron(FeatureTransformer_t ft) {
    SimplePerceptron_t p = (SimplePerceptron_t) malloc(sizeof(struct SimplePerceptron_st));

    check_mem(p);


    p->best_numit = 0;
    p->best_w = NULL;
    p->w = NULL;
    p->w_avg = NULL;
    p->w_beta = NULL;
    p->sv_d = NULL;
    p->instarr_d = NULL;
    p->instarr_pre_d = NULL;
    
    if (ft != NULL)
        EPARSE_CHECK_RETURN(newInitializedGPUMatrix( &(p->instarr_d), "Transformed input on GPU",1, 1, matrixInitNone, NULL, NULL))
    
    
    p->result_d  =NULL;
    p->c = 1;
    p->ft = ft;

    return p;


    error:
    exit(EXIT_FAILURE);
}

eparseError_t scoreSimplePerceptron(SimplePerceptron_t kp, Vector_t inst, bool avg, float *s) {
    Vector_t r = NULL;
    EPARSE_CHECK_RETURN(scoreBatchSimplePerceptron(kp, inst, avg, &r))
    
    *s = (r->data)[0];
    
    return eparseSucess;
}

eparseError_t scoreBatchSimplePerceptron(SimplePerceptron_t kp, Matrix_t instarr, bool avg, Vector_t *result) {
    float zero = 0.f;
    
    Vector_t weight = avg ? (kp->w_avg) : (kp->w);
    
    if ( weight != NULL){
        
        newInitializedCPUVector(result, "result on cpu",instarr->ncol, matrixInitNone, NULL, NULL)
        long nleft = instarr->ncol;
        long offset = 0;
        
        while (nleft > 0) {
                       
            if (kp->ft != NULL){
                EPARSE_CHECK_RETURN(mtrxcolcpy(&( kp->instarr_pre_d ), memoryGPU, instarr, "instarr GPU batch", offset, MIN(nleft, BATCH_SIZE)))
                EPARSE_CHECK_RETURN( transformBatch(kp->ft, kp->instarr_pre_d, &(kp->instarr_d)))
            }
            else
                EPARSE_CHECK_RETURN(mtrxcolcpy(&( kp->instarr_d ), memoryGPU, instarr, "instarr GPU batch", offset, MIN(nleft, BATCH_SIZE)))
            
            newInitializedGPUVector(&(kp->result_d), "result", MIN(nleft, BATCH_SIZE), matrixInitFixed, &zero, NULL)
            
            EPARSE_CHECK_RETURN(prodMatrixVector(kp->instarr_d, true, weight, kp->result_d))
            
            EPARSE_CHECK_RETURN(matrixDatacpyAnyToAny(*result, offset, kp->result_d, 0,  MIN(nleft, BATCH_SIZE) * sizeof(float)));
            
            offset +=  MIN(nleft, BATCH_SIZE);
            nleft -= MIN(nleft, BATCH_SIZE);
        }
        
    }else {
        newInitializedCPUVector(result, "result", instarr->ncol, matrixInitFixed, &zero, NULL)
    }
    
    return eparseSucess;
}

eparseError_t updateSimplePerceptron(SimplePerceptron_t kp, Vector_t sv, long svidx, float change) {
    float zero = 0.f;


    if (kp->w == NULL) {

        newInitializedGPUVector(&(kp->w), "weight", sv->n, matrixInitFixed, &zero, NULL);

        newInitializedGPUVector(&(kp->w_avg), "avg weight", sv->n, matrixInitFixed, &zero, NULL);
        newInitializedGPUVector(&(kp->w_beta), "weight-beta", sv->n, matrixInitFixed, &zero, NULL);

        log_info("w,w-avg and w-beta all intialized as 0-vector of %ld length", sv->n);
    }

    if (sv->dev == memoryGPU) {
        cuda_saxpy(sv->n, change, sv->data, 1, kp->w->data, 1);
        cuda_saxpy(sv->n, change * kp->c, sv->data, 1, kp->w_beta->data, 1);
    } else {
        EPARSE_CHECK_RETURN(cloneVector(&(kp->sv_d), memoryGPU, sv, "device sv"));

        cuda_saxpy(sv->n, change, kp->sv_d->data, 1, kp->w->data, 1);
        cuda_saxpy(sv->n, change * kp->c, kp->sv_d->data, 1, kp->w_beta->data, 1);
    }

    return eparseSucess;
}

eparseError_t dumpSimplePerceptron(FILE *fp, SimplePerceptron_t kp) {
    // TODO: Implement
    return eparseColumnNumberMissmatch;
}

eparseError_t loadSimplePerceptron(FILE *fp, void **kp) {
    // TODO: Implement
    return eparseColumnNumberMissmatch;
}

eparseError_t recomputeSimplePerceptronAvgWeight(SimplePerceptron_t p) {
    
    EPARSE_CHECK_RETURN(cloneVector(&(p->w_avg), memoryGPU, p->w, "w-avg clone of w"))
    
    EPARSE_CHECK_RETURN(cuda_saxpy( p->w->n, -1./(p->c), p->w_beta->data,1,p->w_avg->data,1 ))
    
    return eparseSucess;
}

eparseError_t snapshotBestSimplePerceptron(SimplePerceptron_t sp) {
    debug("Best model snapshot started");

    EPARSE_CHECK_RETURN(cloneMatrix(&(sp->best_w), memoryCPU, sp->w_avg, "Best w-avg"))

    sp->best_numit = 0; //TODO: Fix it

    debug("Best model snapshot completed");

    return eparseSucess;
}
