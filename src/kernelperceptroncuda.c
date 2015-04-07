#include "kernelperceptron.h"
#include "epblas/epcudautil.h"
#include "epblas/epcudakernel.h"

#define BATCH_SIZE 800	//Given that majority of the sentences are shorter than 40 words.


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
    
    kp->t_instBatch = NULL;
    kp->t_yBatch = NULL;
	kp->t_yPowerBatch = NULL;
	kp->t_result = NULL;
	kp->t_y = NULL;
	kp->t_yPower = NULL;
 	kp->t_inst = NULL;

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

	log_info("Dropping kernel perceptron");
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
    
    deleteMatrix( kp->t_instBatch);
    deleteMatrix( kp->t_yBatch);
	deleteMatrix( kp->t_yPowerBatch);
	deleteVector( kp->t_result );

	deleteVector ( kp->t_y);
	deleteVector ( kp->t_yPower);
	deleteVector ( kp->t_inst); 	


    if (kp->kerneltype == POLYNOMIAL_KERNEL)
        deletePolynomialKernelPerceptron((PolynomialKernelPerceptron_t) kp->pDerivedObj);

    free(kp);

    return eparseSucess;

}


	
/**
	Update Perceptron by hypotesis vector
	
	@param kp Kernel Perceptron model
	@param sv New or existing support vector instance
	@param svidx Index of hypothesis vector
	@param change change on hv/sv weight.
*/
eparseError_t updateKernelPerceptron(KernelPerceptron_t kp, Vector_t sv, long svidx, float change) {
	/*
		TODO Gather statistics on the number of sv added and updated.
	*/
    if (kp->kernel->matrix != NULL) {
        if (svidx < kp->kernel->matrix->ncol) {
        	debug("Update alpha for h-vector %ld", svidx);
			
			EPARSE_CHECK_RETURN(updateCudaArrayByIndex(svidx, kp->kernel->alpha->data, change))
			
			EPARSE_CHECK_RETURN(updateCudaArrayByIndex(svidx, kp->kernel->beta->data, change* kp->c))

            return eparseSucess;
        }
        else if (svidx == kp->kernel->matrix->ncol) {
        	debug("Add another h-vector");

            EPARSE_CHECK_RETURN(hstack(&(kp->kernel->matrix), memoryGPU, "kernel matrix", sv, false, false))
				
			EPARSE_CHECK_RETURN(vappend(&(kp->kernel->alpha), memoryGPU, "alpha vector", change))
			EPARSE_CHECK_RETURN(vappend(&(kp->kernel->beta), memoryGPU, "beta vector", change * kp->c))
				
            return eparseSucess;
        }
        else {
            log_err("Update request for sv %ld violates maximum number of %ld", svidx, kp->kernel->matrix->ncol);

            return eparseIndexOutofBound;
        }
    }
    else {
	    debug("Add the first h-vector");
        
        EPARSE_CHECK_RETURN(hstack(&(kp->kernel->matrix), memoryGPU, "kernel matrix", sv, false, false))
        
		EPARSE_CHECK_RETURN(vappend(&(kp->kernel->alpha), memoryGPU, "alpha vector", change))
		EPARSE_CHECK_RETURN(vappend(&(kp->kernel->beta), memoryGPU, "beta vector", change * kp->c))
        
        return eparseSucess;
    }
}


eparseError_t scoreBatchKernelPerceptron(KernelPerceptron_t kp, Matrix_t instarr, bool avg, Vector_t *result) {
	
    check(instarr != NULL, "instarr should be initialized");
    
    float zero = 0.f;
	
	Matrix_t kernel_matrix = kp->kernel->matrix;
	
    if (kernel_matrix != NULL) {
        if (kp->kerneltype == POLYNOMIAL_KERNEL) {

            PolynomialKernelPerceptron_t pkp = (PolynomialKernelPerceptron_t) kp->pDerivedObj;
			
			//log_info("so");
			newInitializedCPUVector(result, "result",instarr->ncol, matrixInitNone, NULL, NULL)
			
			long nleft = instarr->ncol;
			long offset = 0;
			float zero = 0.f;
			
			while(nleft > 0){
				//log_info("%ld col left", nleft);
				
				EPARSE_CHECK_RETURN(newInitializedGPUMatrix(&(kp->t_yBatch), "t_yBatch", kernel_matrix->ncol, MIN(nleft, BATCH_SIZE), matrixInitFixed, &(pkp->bias), NULL))


				EPARSE_CHECK_RETURN(mtrxcolcpy(&(kp->t_instBatch), memoryGPU, instarr, "t_instBatch", offset, MIN(nleft, BATCH_SIZE)))
				EPARSE_CHECK_RETURN(prodMatrixMatrix(kernel_matrix, true, kp->t_instBatch, kp->t_yBatch))
				
				EPARSE_CHECK_RETURN(powerMatrix(kp->t_yBatch, pkp->power, NULL))
        
        
		        newInitializedGPUVector(&(kp->t_result), "t_result", MIN(nleft, BATCH_SIZE), matrixInitFixed,&zero, NULL)


		        if (avg) 
		        	EPARSE_CHECK_RETURN(prodMatrixVector(kp->t_yBatch, true, kp->kernel->alpha_avg, kp->t_result))
		        else 
		        	EPARSE_CHECK_RETURN(prodMatrixVector(kp->t_yBatch, true, kp->kernel->alpha, kp->t_result))
							
				EPARSE_CHECK_RETURN(matrixDatacpyAnyToAny(*result, offset, kp->t_result, 0,  MIN(nleft, BATCH_SIZE) * sizeof(float)));
				
				offset +=  MIN(nleft, BATCH_SIZE);
				nleft -= MIN(nleft, BATCH_SIZE);
			}			
        } 
		else {
			return eparseKernelType;
		}
				
    }else{
    	newInitializedCPUVector(result, "result", instarr->ncol,  matrixInitFixed, &zero, NULL)
    }

    return eparseSucess;

error:
    return eparseMemoryAllocationError;
}




eparseError_t scoreKernelPerceptron(KernelPerceptron_t kp, Vector_t inst, bool avg, float *s) {
    *s = 0.0;

    Matrix_t kernel_matrix = kp->kernel->matrix;
    if (kernel_matrix != NULL) {
        if (kp->kerneltype == POLYNOMIAL_KERNEL) {

            PolynomialKernelPerceptron_t pkp = (PolynomialKernelPerceptron_t) kp->pDerivedObj;

            newInitializedGPUVector(&(kp->t_y), "t_y", kernel_matrix->ncol, matrixInitFixed, &(pkp->bias), NULL)

		    if (inst->dev == memoryGPU){
				
            	EPARSE_CHECK_RETURN(prodMatrixVector(kernel_matrix, true, inst, kp->t_y))
	 		}
	 		else{
				EPARSE_CHECK_RETURN(cloneVector(&(kp->t_inst), memoryGPU,inst, "instance on GPU" ))
            	EPARSE_CHECK_RETURN(prodMatrixVector(kernel_matrix, true, kp->t_inst, kp->t_y))
		
			}

            //newInitializedGPUVector(&(kp->t_yPower), "t_yPower", kernel_matrix->ncol, matrixInitNone, NULL, NULL)

            EPARSE_CHECK_RETURN(powerMatrix(kp->t_y, pkp->power, NULL))
            
        } else {
            //TODO: Handle unknown subtype.
        }


        if (avg)
            EPARSE_CHECK_RETURN(dot(kp->kernel->alpha_avg, kp->t_y, s))
        else
            EPARSE_CHECK_RETURN(dot(kp->kernel->alpha, kp->t_y, s))
            
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

    fprintf(fp, "nsv=%lu\n", kp->best_kernel->matrix->ncol );
    fprintf(fp, "edim=%lu\n", kp->best_kernel->matrix->nrow);
    fprintf(fp, "numit=%d\n", kp->best_numit);
    fprintf(fp, "c=%d\n", kp->c);


    for (long i = 0; i < kp->best_kernel->matrix->ncol; i++) {
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

		
	Vector_t temp_avg = NULL;
	newInitializedCPUVector(&(temp_avg), "Temp weight vector", nsv, matrixInitNone, NULL, NULL)

    int real_idx;
    for (int i = 0; i < nsv; i++) {
        n = fscanf(fp, "alpha[%d]=%f\n", &real_idx, &((temp_avg->data)[i]));

        check(n == 2, "Either index (%d) or coefficient(%f) is missing", real_idx, (temp_avg->data)[i]);
        check(i == real_idx, "Expected index (%d) does not match with current index(%d)", i, real_idx);
    }
	
	EPARSE_CHECK_RETURN(cloneVector(&(model->kernel->alpha), memoryGPU, temp_avg, "sv weight"))
	EPARSE_CHECK_RETURN(cloneVector(&(model->kernel->alpha_avg), memoryGPU, temp_avg, "sv avg weight"))		
		
	deleteVector(temp_avg);

    model->kernel->matrix = NULL;
	Matrix_t kernel_matrix_temp = NULL;

    EPARSE_CHECK_RETURN(newInitializedCPUMatrix(&(kernel_matrix_temp), "temp kernel matrix", edim, nsv, matrixInitNone, NULL , NULL))

    long real_lidx;
    for (long i = 0; i < kernel_matrix_temp->n; i++) {

        n = fscanf(fp, "K[%ld]=%f\n", &real_lidx, &((kernel_matrix_temp->data)[i]));

        check(n == 2, "Either index (%ld) or coefficient(%f) is missing", real_lidx, (kernel_matrix_temp->data)[i]);
        check(i == real_lidx, "Expected index (%ld) does not match with current index(%ld)", i, real_lidx);
    }
	
	EPARSE_CHECK_RETURN(cloneMatrix(&(model->kernel->matrix), memoryGPU, kernel_matrix_temp, "kernel matrix"))
		
	deleteMatrix(kernel_matrix_temp);

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
	
	Vector_t alpha_host = NULL, alpha_avg_host = NULL, beta_host = NULL;
	

    EPARSE_CHECK_RETURN(cloneVector(&(alpha_host), memoryCPU, kp->kernel->alpha, "temp alpha on host"))
	EPARSE_CHECK_RETURN(cloneVector(&(beta_host), memoryCPU, kp->kernel->beta, "temp beta on host"))
		
	newInitializedCPUVector(&(alpha_avg_host), "sv avg. alpha on host", kp->kernel->matrix->ncol, matrixInitNone, NULL, NULL)	

    for (long i = 0; i < kp->kernel->matrix->ncol; i++) {

        (alpha_avg_host->data)[i] = (alpha_host->data)[i] - (beta_host->data)[i] / (kp->c);

    }
	
	EPARSE_CHECK_RETURN(cloneVector(&(kp->kernel->alpha_avg), memoryGPU, alpha_avg_host, "sv avg. alpha"))
		
		
	deleteVector(alpha_host)
	deleteVector(alpha_avg_host)	
	deleteVector(beta_host)

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
