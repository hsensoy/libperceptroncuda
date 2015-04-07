#include <stdio.h>
#include <stdlib.h>
#include "epblas/epblas.h"

/*
 * CUnit Test Suite
 */

Matrix_t A=NULL;
Matrix_t A_host=NULL;

Vector_t x= NULL;
Vector_t x_host= NULL;

Vector_t y = NULL;
Vector_t y_host = NULL;
Vector_t ones=NULL;

float zero=0.,one=1.,two=2.,three,result;


void testMatrixVectorProduct() {

    // Dot product 1
//    newInitializedGPUMatrix(&A, "matrix A", 2, 2, matrixInitNone, NULL, NULL);
//    newInitializedGPUVector(&x, "vector x", 2, matrixInitNone, NULL, NULL);
    newInitializedGPUVector(&y, "vector y", 3, matrixInitNone, NULL, NULL);
	
	
	newInitializedCPUMatrix(&A_host, "matrix A on host", 3, 3, matrixInitNone, NULL, NULL);	
	(A_host->data)[0] = 1;
	(A_host->data)[1] = 4;
	(A_host->data)[2] = 7;
	(A_host->data)[3] = 2;
	(A_host->data)[4] = 5;
	(A_host->data)[5] = 8;
	(A_host->data)[6] = 3;
	(A_host->data)[7] = 6;
	(A_host->data)[8] = 9;
	
	newInitializedCPUVector(&x_host, "vector x on host", 3, matrixInitNone, NULL, NULL);	
	(x_host->data)[0] = 1;
	(x_host->data)[1] = 2;
	(x_host->data)[2] = 3;
	
	EPARSE_CHECK_RETURN(cloneMatrix(&A, memoryGPU, A_host, "matrix A on GPU"))
	EPARSE_CHECK_RETURN(cloneMatrix(&x, memoryGPU, x_host, "vector x on GPU"))
		
	EPARSE_CHECK_RETURN(prodMatrixVector(A, false, x, y))
		
	EPARSE_CHECK_RETURN(cloneMatrix(&y_host, memoryCPU, y, "vector y on CPU"))
		
	check((y_host->data)[0] == 14.,"Expected value at index %d was %f where %f found", 0, 14., (y_host->data)[0])
	check((y_host->data)[1] == 32.,"Expected value at index %d was %f where %f found", 1, 32., (y_host->data)[1])
	check((y_host->data)[2] == 50.,"Expected value at index %d was %f where %f found", 2, 51., (y_host->data)[2])
		
	log_info("Try with transpose");
	EPARSE_CHECK_RETURN(prodMatrixVector(A, true, x, y))
	
	EPARSE_CHECK_RETURN(cloneMatrix(&y_host, memoryCPU, y, "vector y on CPU"))
	
	check((y_host->data)[0] == 30.,"Expected value at index %d was %f where %f found", 0, 30., (y_host->data)[0])
	check((y_host->data)[1] == 36.,"Expected value at index %d was %f where %f found", 1, 36., (y_host->data)[1])
	check((y_host->data)[2] == 42.,"Expected value at index %d was %f where %f found", 2, 42., (y_host->data)[2])

	log_info("Retry without transpose...");
	EPARSE_CHECK_RETURN(prodMatrixVector(A, false, x, y))

	EPARSE_CHECK_RETURN(cloneMatrix(&y_host, memoryCPU, y, "vector y on CPU"))

	check((y_host->data)[0] == 14.,"Expected value at index %d was %f where %f found", 0, 14., (y_host->data)[0])
	check((y_host->data)[1] == 32.,"Expected value at index %d was %f where %f found", 1, 32., (y_host->data)[1])
	check((y_host->data)[2] == 50.,"Expected value at index %d was %f where %f found", 2, 51., (y_host->data)[2])
		
	log_info("Init with epblas");
	float one = 2.;
	
	int N = 8000	;
	
	/*	
	newInitializedCPUMatrix(&A_host, "matrix A on host", N, N, matrixInitNone, NULL, NULL);	
	for(size_t i = 0; i < N*N; ++i)
	{
			(A_host->data)[i] = one;
	}
	
	A = NULL;
	EPARSE_CHECK_RETURN(cloneMatrix(&A, memoryGPU, A_host, "matrix A on GPU"))
		*/
	
    newInitializedGPUVector(&x, "vector x", N, matrixInitFixed, &one, NULL);
	newInitializedGPUVector(&y, "vector y", N, matrixInitNone, NULL, NULL);
	
	newInitializedGPUMatrix(&A, "matrix A", N, N, matrixInitFixed, &one, NULL);	
	
	
//	EPARSE_CHECK_RETURN(cloneMatrix(&A_host, memoryCPU, A, "vector y on CPU"))
		
//		for(int i = 0; i < N * N; ++i)
//			check((A_host->data)[i] == one,"Expected value at index %d was %f where %f found", i, one, (y_host->data)[0])
				

    EPARSE_CHECK_RETURN(prodMatrixVector(A, false, x, y))
		
	EPARSE_CHECK_RETURN(cloneMatrix(&y_host, memoryCPU, y, "vector y on CPU"))
		
	for(int i = 0; i < N; ++i)
		check((y_host->data)[i] == (float)(N*one*one),"Expected value at index %d was %f where %f found", i, N*one*one*1., (y_host->data)[i])
			
	
	exit(EXIT_SUCCESS);

error:
	exit(EXIT_FAILURE);
}


int main() {
    testMatrixVectorProduct();
}
