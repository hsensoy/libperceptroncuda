#include "perceptron.h"

#include "kernelperceptron.h"
#include "simpleperceptron.h"

eparseError_t setPerceptronParallism(int nslave) {

    return eparseInvalidOperationRequest;
}


eparseError_t getPerceptronMaxParallism(int *nslave) {
    //TODO: Maybe return the number of CUDA cores :)

    return eparseInvalidOperationRequest;
}

eparseError_t getPerceptronDynamicParallism(bool *dynamic) {

    return eparseInvalidOperationRequest;
}

static Perceptron_t newPerceptron(enum PerceptronType type){
    Perceptron_t p = (Perceptron_t) malloc(sizeof(struct Perceptron_st));
    check_mem(p);

    p->type = type;

    return p;

    error:
    exit(EXIT_FAILURE);
}

Perceptron_t newPolynomialKernelPerceptron(int power, float bias){
    Perceptron_t p = newPerceptron(KERNEL_PERCEPTRON);

    p->pDeriveObj = (void *)__newPolynomialKernelPerceptron(power,bias);


    return p;
}

Perceptron_t newSimplePerceptron(FeatureTransformer_t ft){
    Perceptron_t p = newPerceptron(SIMPLE_PERCEPTRON);

    p->pDeriveObj = (void *)__newSimplePerceptron(ft);

    return p;
}

eparseError_t deletePerceptron(Perceptron_t p){

    switch(p->type){
        case KERNEL_PERCEPTRON:
            deleteKernelPerceptron((KernelPerceptron_t)(p->pDeriveObj));

            break;
        case SIMPLE_PERCEPTRON:
            deleteSimplePerceptron((SimplePerceptron_t)(p->pDeriveObj));
            break;

        default:
            return eparseKernelType;
    }

    free(p);

    return eparseSucess;
}

eparseError_t score(Perceptron_t p, Vector_t inst, bool avg, float *s){

    switch(p->type){
        case KERNEL_PERCEPTRON:
            return scoreKernelPerceptron((KernelPerceptron_t)(p->pDeriveObj), inst, avg, s);
        case SIMPLE_PERCEPTRON:
            return scoreSimplePerceptron((SimplePerceptron_t)(p->pDeriveObj), inst, avg, s);
        default:
            return eparseKernelType;
    }
}


eparseError_t scoreBatch(Perceptron_t p, Matrix_t instarr, bool avg, Vector_t *result){

    switch(p->type){
        case KERNEL_PERCEPTRON:
            return scoreBatchKernelPerceptron((KernelPerceptron_t)(p->pDeriveObj), instarr, avg, result);
        case SIMPLE_PERCEPTRON:
            return scoreBatchSimplePerceptron((SimplePerceptron_t)(p->pDeriveObj), instarr, avg, result);
        default:
            return eparseKernelType;
    }

}

eparseError_t update(Perceptron_t p, Vector_t sv, long svidx, float change){

    switch(p->type){
        case KERNEL_PERCEPTRON:
            return updateKernelPerceptron((KernelPerceptron_t)(p->pDeriveObj), sv, svidx, change);
        case SIMPLE_PERCEPTRON:
            return updateSimplePerceptron((SimplePerceptron_t)(p->pDeriveObj), sv, svidx, change);
        default:
            return eparseKernelType;
    }

}

eparseError_t dumpPerceptronModel(FILE *fp, Perceptron_t p){

    fprintf(fp, "type=%d\n", p->type);

    switch(p->type){
        case KERNEL_PERCEPTRON:
            return dumpKernelPerceptron(fp, (KernelPerceptron_t)(p->pDeriveObj));
        case SIMPLE_PERCEPTRON:
            return dumpSimplePerceptron(fp, (SimplePerceptron_t)(p->pDeriveObj));
        default:
            return eparseKernelType;
    }

}

eparseError_t loadPerceptronModel(FILE *fp, Perceptron_t *p){
    enum PerceptronType subtype;

    int n = fscanf(fp, "type=%d\n", &subtype);

    debug("Perceptrin type is %d", subtype);

    //check(n == 1, "No kernel type found in file");

    switch(subtype){
        case KERNEL_PERCEPTRON:
            *p = newPerceptron(KERNEL_PERCEPTRON);

            return loadKernelPerceptron(fp, &((*p)->pDeriveObj));
        case SIMPLE_PERCEPTRON:
            *p = newPerceptron(SIMPLE_PERCEPTRON);

            return loadSimplePerceptron(fp, &((*p)->pDeriveObj));
        default:
            return eparseKernelType;
    }
}

eparseError_t showStats(Perceptron_t p){

    switch(p->type){
        case KERNEL_PERCEPTRON:
            return showStats((KernelPerceptron_t)p->pDeriveObj);
        case SIMPLE_PERCEPTRON:
            return showStats((SimplePerceptron_t)p->pDeriveObj);

        default:
            return eparseKernelType;
    }
}


eparseError_t snapshotBest(Perceptron_t p){
    switch(p->type){
        case KERNEL_PERCEPTRON:
            return snapshotBestKernelPerceptron((KernelPerceptron_t)p->pDeriveObj);
        case SIMPLE_PERCEPTRON:
            return snapshotBestSimplePerceptron((SimplePerceptron_t)p->pDeriveObj);

        default:
            return eparseKernelType;
    }
}


eparseError_t recomputeAvgWeight(Perceptron_t p){
    switch(p->type){
        case KERNEL_PERCEPTRON:
            return recomputeKernelPerceptronAvgWeight((KernelPerceptron_t)p->pDeriveObj);
        case SIMPLE_PERCEPTRON:
            return recomputeSimplePerceptronAvgWeight((SimplePerceptron_t)p->pDeriveObj);

        default:
            return eparseKernelType;
    }
}