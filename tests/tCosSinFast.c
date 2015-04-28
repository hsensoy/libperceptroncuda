#include <stdio.h>
#include <stdlib.h>
#include "epblas/epblas.h"
#include "epblas/cudakernel.h"

/*
 * CUnit Test Suite
 */

void testFastCosSinTransform() {
	Matrix_t A=NULL;
	Matrix_t R=NULL;
	Matrix_t Rfast=NULL;
	
	Matrix_t R_host=NULL;
	Matrix_t Rfast_host=NULL;

	EPARSE_CHECK_RETURN(newInitializedGPUMatrix(&A, "Matrix A", 1000,1000, matrixInitNone, NULL, NULL))
	
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

	curandSetPseudoRandomGeneratorSeed(gen, 777);

	curandStatus_t status = curandGenerateNormal(gen,A->data, A->n, 0.0, 1.f);

	if (status != CURAND_STATUS_SUCCESS){
		log_info("Error in CUDA random number generation %d", status);

		return eparseCUDAError;
	}

	curandDestroyGenerator(gen);
	
	EPARSE_CHECK_RETURN(newInitializedGPUMatrix(&R, "Matrix R", 2000,1000, matrixInitNone, NULL, NULL))
	EPARSE_CHECK_RETURN(newInitializedGPUMatrix(&Rfast, "Matrix R(fasy)", 2000,1000, matrixInitNone, NULL, NULL))
	
	EPARSE_CHECK_RETURN(vsCosSinMatrix(1000,1000,A->data,R->data))
	EPARSE_CHECK_RETURN(vsCosSinMatrixFast(1000,1000,A->data,Rfast->data))
	
	
	EPARSE_CHECK_RETURN(cloneMatrix(&R_host, memoryCPU, R, "R on host"))
	EPARSE_CHECK_RETURN(cloneMatrix(&Rfast_host, memoryCPU, Rfast, "Rfast on host"))
	
	check(R_host->n == Rfast_host->n, "R-host %ld R-fast-host %ld",R_host->n, Rfast_host->n);
	
	for(long i = 0 ; i < R_host->n, i++){
		check( (Rfast_host->data)[i] == (R_host->data)[i], "Fast result %f on index %d does not match with %f",(Rfast_host->data)[i], i, (R_host->data)[i] );
	}
	
	
	exit(EXIT_SUCCESS);
error:
	exit(EXIT_FAILURE);

}

int main() {
	testFastCosSinTransform();
}
