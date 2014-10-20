#include <CoreFoundation/CoreFoundation.h>
#include "kernelperceptron.h"

/**
* Create a new kernel structure
*
* @return new kernel structure.
*/
Kernel_t newKernel() {
    Kernel_t k = (Kernel_t) malloc(sizeof(struct Kernel_st));

    check(k != NULL, "Kernel_t structure allocation error");

    //EPARSE_CHECK_RETURN(newInitializedMatrix(&(k->matrix), memoryCPU, "kernel matrix", 0, 1, matrixInitNone, NULL, NULL))
    //EPARSE_CHECK_RETURN(newInitializedMatrix(&(k->alpha), memoryCPU, "sv weight", 0, 1, matrixInitNone, NULL, NULL))
    //EPARSE_CHECK_RETURN(newInitializedMatrix(&(k->beta), memoryCPU, NULL, 0, 1, matrixInitNone, NULL, NULL))
    //EPARSE_CHECK_RETURN(newInitializedMatrix(&(k->alpha_avg), memoryCPU, "sv avg weight", 0, 1, matrixInitNone, NULL, NULL))

    k->matrix = NULL;
    k->alpha = NULL;
    k->alpha_avg = NULL;
    k->beta = NULL;

    return k;
    error:
    exit(1);
}

static KernelPerceptron_t newKernelPerceptron(enum KernelType kerneltype) {
    KernelPerceptron_t kp = (KernelPerceptron_t) malloc(sizeof(struct KernelPerceptron_st));
    check(kp != NULL, "KernelPerceptron allocation error");

    kp->best_kernel = newKernel();
    kp->kernel = newKernel();

    kp->kerneltype = kerneltype;
    kp->c = 1;

    return kp;

    error:
    exit(1);
}



/**
* Free kernel structure.
*
* @param k Kernel structure
*/
void deleteKernel(Kernel_t k) {
    if (k != NULL) {

        deleteMatrix(k->matrix);
        deleteVector(k->alpha);
        deleteVector(k->beta);
        deleteVector(k->alpha_avg);

        //      deleteKernelIndex(k->idx);
        free(k);
    }
}

eparseError_t deletePolynomialKernelPerceptron(PolynomialKernelPerceptron_t pkp) {
    free(pkp);

    return eparseSucess;
}

eparseError_t deleteKernelPerceptron(KernelPerceptron_t kp) {

    deleteKernel(kp->kernel);
    deleteKernel(kp->best_kernel);

    if (kp->kerneltype == POLYNOMIAL_KERNEL)
        deletePolynomialKernelPerceptron((PolynomialKernelPerceptron_t) kp->pDerivedObj);

    free(kp);

    return eparseSucess;

}


eparseError_t updateKernelPerceptron(KernelPerceptron_t kp, Vector_t sv, long svidx, float change) {
    Vector_t v = NULL;

    if (kp->kernel->matrix != NULL) {
        if (svidx < kp->kernel->matrix->nrow) {


            EPARSE_CHECK_RETURN(cloneVector(&v, memoryCPU, kp->kernel->alpha, "temp v"))

            (v->data)[svidx] += change;

            EPARSE_CHECK_RETURN(cloneVector(&(kp->kernel->alpha), memoryGPU, v, "alpha vector"))

            EPARSE_CHECK_RETURN(cloneVector(&v, memoryCPU, kp->kernel->beta, "temp v"))

            (v->data)[svidx] += change * kp->c;

            EPARSE_CHECK_RETURN(cloneVector(&(kp->kernel->beta), memoryGPU, v, "beta vector"))

            deleteVector(v);

            return eparseSucess;
        }
        else if (svidx == kp->kernel->matrix->nrow) {

            float init = change;

            newInitializedCPUVector(&v, "temp v", 1, matrixInitFixed, &init, NULL)

            EPARSE_CHECK_RETURN(vstackMatrix(&(kp->kernel->matrix), memoryGPU, "kernel matrix", sv, true, false))
            EPARSE_CHECK_RETURN(vstackMatrix(&(kp->kernel->alpha), memoryGPU, "alpha vector", v, true, false))

            init = change * kp->c;
            newInitializedCPUVector(&v, "temp v", 1, matrixInitFixed, &init, NULL)

            EPARSE_CHECK_RETURN(vstackMatrix(&(kp->kernel->beta), memoryGPU, "beta vector", v, true, false))


            deleteVector(v);

            return eparseSucess;
        }
        else {

            log_err("Update request for sv %ld violates maximum number of %ld", svidx, kp->kernel->matrix->nrow);
            return eparseIndexOutofBound;
        }
    }
    else {
        float init = change;

        newInitializedCPUVector(&v, "temp v", 1, matrixInitFixed, &init, NULL)

        EPARSE_CHECK_RETURN(vstackMatrix(&(kp->kernel->matrix), memoryGPU, "kernel matrix", sv, true, false))
        EPARSE_CHECK_RETURN(vstackMatrix(&(kp->kernel->alpha), memoryGPU, "alpha vector", v, true, false))

        init = change * kp->c;
        newInitializedCPUVector(&v, "temp v", 1, matrixInitFixed, &init, NULL)

        EPARSE_CHECK_RETURN(vstackMatrix(&(kp->kernel->beta), memoryGPU, "beta vector", v, true, false))

        deleteVector(v);

        return eparseSucess;
    }
}





static Matrix_t Y = NULL;
static Matrix_t YPower = NULL;
static Vector_t result = NULL;

eparseError_t scoreBatchKernelPerceptron(KernelPerceptron_t kp, Matrix_t instarr, bool avg, Vector_t *result) {

    check(instarr != NULL, "instarr should be initialized");


    float zero = 0;


    newInitializedCPUVector(result, "result", instarr->nrow, matrixInitFixed, &zero, NULL)

    Matrix_t kernel_matrix = kp->kernel->matrix;
    if (kernel_matrix != NULL) {
        if (kp->kerneltype == POLYNOMIAL_KERNEL) {

            PolynomialKernelPerceptron_t pkp = (PolynomialKernelPerceptron_t) kp->pDerivedObj;


            EPARSE_CHECK_RETURN(newInitializedGPUMatrix(&Y, "Y", kernel_matrix->nrow, instarr->nrow, matrixInitFixed, &(pkp->bias), NULL))

            Matrix_t instarr_device = NULL;

            EPARSE_CHECK_RETURN(cloneMatrix(&instarr_device, memoryGPU, instarr, "Device clone of instarr"))

            EPARSE_CHECK_RETURN(prodMatrixMatrix(kernel_matrix, instarr_device, true, Y))

            EPARSE_CHECK_RETURN(deleteMatrix(instarr_device))

            EPARSE_CHECK_RETURN(newInitializedGPUMatrix(&YPower, "Y Power", kernel_matrix->nrow, instarr->nrow, matrixInitNone, NULL, NULL))

            EPARSE_CHECK_RETURN(powerMatrix(Y, pkp->power, YPower))
        } else {
            //TODO: Handle unknow subtype.
        }


        if (avg) EPARSE_CHECK_RETURN(prodMatrixVector(YPower, true, kp->kernel->alpha_avg, *result))
        else EPARSE_CHECK_RETURN(prodMatrixVector(YPower, true, kp->kernel->alpha, *result))
    }


    return eparseSucess;

    error:
    return eparseMemoryAllocationError;
}

static Vector_t y = NULL;
static Vector_t yPower = NULL;

eparseError_t scoreKernelPerceptron(KernelPerceptron_t kp, Vector_t inst, bool avg, float *s) {
    *s = 0.0;

    Matrix_t kernel_matrix = kp->kernel->matrix;
    if (kernel_matrix != NULL) {
        if (kp->kerneltype == POLYNOMIAL_KERNEL) {

            PolynomialKernelPerceptron_t pkp = (PolynomialKernelPerceptron_t) kp->pDerivedObj;

            newInitializedGPUVector(&y, "y", kernel_matrix->nrow, matrixInitFixed, &(pkp->bias), NULL)

            EPARSE_CHECK_RETURN(prodMatrixVector(kernel_matrix, false, inst, y))

            newInitializedGPUVector(&yPower, "y Power", kernel_matrix->nrow, matrixInitNone, NULL, NULL)

            EPARSE_CHECK_RETURN(powerMatrix(y, pkp->power, yPower))
        } else {
            //TODO: Handle unknown subtype.
        }


        if (avg)
            EPARSE_CHECK_RETURN(dot(kp->kernel->alpha_avg, yPower, s))
        else
            EPARSE_CHECK_RETURN(dot(kp->kernel->alpha, yPower, s))
    }

    return eparseSucess;
}

KernelPerceptron_t __newPolynomialKernelPerceptron(int power, float bias) {
    KernelPerceptron_t kp = newKernelPerceptron(POLYNOMIAL_KERNEL);


    PolynomialKernelPerceptron_t pkp = (PolynomialKernelPerceptron_t) malloc(sizeof(struct PolynomialKernelPerceptron_st));

    check(pkp != NULL, "PolynomialKernelPerceptron_t allocation error");

    //EPARSE_CHECK_RETURN(newSupportVectorIndex(&(kp->svIdx), MAX_TRAINING_SENTENCE));

    pkp->bias = bias;
    pkp->power = power;


    kp->pDerivedObj = (void*)pkp;


    log_info("Polynomial kernel of degree %d with bias %f is created", pkp->power, pkp->bias);


    return kp;


    error:
    exit(1);
}


eparseError_t dumpKernelPerceptron(FILE *fp, KernelPerceptron_t kp) {

    fprintf(fp, "kernel=%d\n", kp->kerneltype);
    fprintf(fp, "power=%d\n", ((PolynomialKernelPerceptron_t)kp->pDerivedObj)->power);
    fprintf(fp, "bias=%f\n", ((PolynomialKernelPerceptron_t)kp->pDerivedObj)->bias);

    fprintf(fp, "nsv=%lu\n", kp->best_kernel->matrix->nrow );
    fprintf(fp, "edim=%lu\n", kp->best_kernel->matrix->ncol);
    fprintf(fp, "numit=%d\n", kp->best_numit);
    fprintf(fp, "c=%d\n", kp->c);


    for (long i = 0; i < kp->best_kernel->matrix->nrow; i++) {
        //fprintf(fp, "alpha_avg[%lu]=%f alpha[%lu]=%f beta[%lu]=%f\n", i, (kp->best_alpha_avg)[i],i, (kp->alpha)[i],i, (kp->beta)[i]);
        fprintf(fp, "alpha[%lu]=%f\n", i, (kp->best_kernel->alpha_avg->data)[i]);
    }

    for (size_t i = 0; i < kp->best_kernel->matrix->n; i++) {
        fprintf(fp, "K[%lu]=%f\n", i, (kp->best_kernel->matrix->data)[i]);
    }

    return eparseSucess;

}

eparseError_t loadKernelPerceptron(FILE *fp, void **kp){
    int n;
    int power;
    float bias;

    enum KernelType kerneltype;

    n = fscanf(fp, "kernel=%d\n", &kerneltype);

    debug("Kernel type is %d", kerneltype);


    n = fscanf(fp, "power=%d\n", &power);
    check(n == 1 && power > 0, "No power found for polynomial model");

    debug("Power is %d", power);
    n = fscanf(fp, "bias=%f\n", &bias);
    check(n == 1, "No bias found for polynomial model");
    debug("Bias is %f", bias);

    KernelPerceptron_t model = __newPolynomialKernelPerceptron(power, bias);


    int nsv, edim;
    n = fscanf(fp, "nsv=%d\nedim=%d\nnumit=%d\nc=%d\n", &nsv, &edim, &(model->best_numit), &(model->c));
    check(n == 4, "Num s.v. or embedding dimension or numit or c is missing in model file");

    debug("Number of Support Vectors are %d", nsv);
    debug("Embedding dimension is %d", edim);
    debug("Number of Iterations are %d", model->best_numit);
    debug("C is %d", model->c);

    model->kernel->alpha = NULL;
    model->kernel->alpha_avg = NULL;

    newInitializedCPUVector(&(model->kernel->alpha), "sv weight", nsv, matrixInitNone, NULL, NULL)
    newInitializedCPUVector(&(model->kernel->alpha_avg), "sv avg weight", nsv, matrixInitNone, NULL, NULL)


    int real_idx;
    for (int i = 0; i < nsv; i++) {
        n = fscanf(fp, "alpha[%d]=%f\n", &real_idx, &((model->kernel->alpha->data)[i]));

        check(n == 2, "Either index (%d) or coefficient(%f) is missing", real_idx, (model->kernel->alpha->data)[i]);
        check(i == real_idx, "Expected index (%d) does not match with current index(%d)", i, real_idx);

        (model->kernel->alpha_avg->data)[i] = (model->kernel->alpha->data)[i];
    }

    model->kernel->matrix = NULL;

    EPARSE_CHECK_RETURN(newInitializedCPUMatrix(&(model->kernel->matrix), "kernel matrix", nsv, edim, matrixInitNone, NULL , NULL))

    long real_lidx;
    for (long i = 0; i < model->kernel->matrix->n; i++) {

        n = fscanf(fp, "K[%ld]=%f\n", &real_lidx, &((model->kernel->matrix->data)[i]));

        check(n == 2, "Either index (%ld) or coefficient(%f) is missing", real_lidx, (model->kernel->matrix->data)[i]);
        check(i == real_lidx, "Expected index (%ld) does not match with current index(%ld)", i, real_lidx);
    }

    *kp = model;

    return eparseSucess;
    error:
    return eparseKernelPerceptronLoadError;
}

eparseError_t showStatsKernelPerceptron(KernelPerceptron_t kp){

    long nsv = kp->kernel->matrix->nrow;

    //log_info("%ld (%f of total %ld) support vectors", nsv, (nsv * 1.) / max_sv, max_sv);
    log_info("%ld support vectors", nsv);

    return eparseSucess;
}

eparseError_t recomputeKernelPerceptronAvgWeight(KernelPerceptron_t kp){

    newInitializedCPUVector(&(kp->kernel->alpha_avg), "sv avg. alpha", kp->kernel->matrix->nrow, matrixInitNone, NULL, NULL)

    //TODO: Hi performance version of this should be possible in BLAS.
#pragma ivdep
    for (long i = 0; i < kp->kernel->matrix->nrow; i++) {

        (kp->kernel->alpha_avg->data)[i] = (kp->kernel->alpha->data)[i] - (kp->kernel->beta->data)[i] / (kp->c);

    }

    return eparseSucess;

}

eparseError_t snapshotBestKernelPerceptron(KernelPerceptron_t kp){
    debug("Best model snapshot started");

    EPARSE_CHECK_RETURN(cloneMatrix(&(kp->best_kernel->matrix), memoryCPU, kp->kernel->matrix, NULL))
    //EPARSE_CHECK_RETURN(cloneVector(&(kp->best_kernel->alpha), memoryCPU, kp->kernel->alpha, NULL))
    EPARSE_CHECK_RETURN(cloneVector(&(kp->best_kernel->alpha_avg), memoryCPU, kp->kernel->alpha_avg, NULL))

    kp->best_numit = 0; //TODO: Fix it

    debug("Best model snapshot completed");

    return eparseSucess;
}