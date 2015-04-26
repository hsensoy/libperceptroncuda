#include "simpleperceptron.h"
#include "epcudakernel.h"

#define BATCH_SIZE 900	//Given that majority of the sentences are shorter than 30 words.

eparseError_t deleteSimplePerceptron(SimplePerceptron_t sp) {
    deleteVector(sp->best_w);
    deleteVector(sp->w);
    deleteVector(sp->w_avg);

    free(sp);

    return eparseSucess;

}

SimplePerceptron_t __newSimplePerceptron() {
    SimplePerceptron_t p = (SimplePerceptron_t) malloc(sizeof(struct SimplePerceptron_st));

    check_mem(p);


    p->best_numit = 0;
    p->best_w = NULL;
    p->w = NULL;
    p->w_avg = NULL;
    p->w_beta = NULL;
    p->sv_d = NULL;
    p->instarr_d = NULL;
    p->result_d  =NULL;
    p->c = 1;

    return p;


    error:
    exit(EXIT_FAILURE);
}

eparseError_t scoreSimplePerceptron(SimplePerceptron_t kp, Vector_t inst, bool avg, float *s) {
    if (avg) {
        if (kp->w_avg == NULL)
            *s = 0.f;
        else {
            if (inst->dev == memoryGPU) {
                EPARSE_CHECK_RETURN(dot(kp->w_avg, inst, s))
            }
            else {
                Vector_t inst_d = NULL;
                EPARSE_CHECK_RETURN(cloneVector(&inst_d, memoryGPU, inst, "device instance"));

                EPARSE_CHECK_RETURN(dot(kp->w_avg, inst_d, s))

                deleteVector(inst_d);
            }
        }
    } else {
        if (kp->w == NULL)
            *s = 0.f;
        else {
            if (inst->dev == memoryGPU) {
                EPARSE_CHECK_RETURN(dot(kp->w, inst, s))
            }
            else {
                Vector_t inst_d = NULL;
                EPARSE_CHECK_RETURN(cloneVector(&inst_d, memoryGPU, inst, "device instance"));

                EPARSE_CHECK_RETURN(dot(kp->w, inst_d, s))

                deleteVector(inst_d);
            }
        }
    }
    
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
