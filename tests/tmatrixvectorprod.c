#include <stdio.h>
#include <stdlib.h>
#include "epblas/epblas.h"

/*
 * CUnit Test Suite
 */

Matrix_t A=NULL;
Matrix_t x= NULL;
Vector_t y = NULL;
Vector_t y_host = NULL;
Vector_t ones=NULL;

float zero=0.,one=1.,two=2.,three,result;


void testMatrixVectorProduct() {

    // Dot product 1
    newInitializedGPUMatrix(&A, "matrix A", 1000, 1000, matrixInitFixed, &one, NULL);
    newInitializedGPUVector(&x, "vector x", 1000, matrixInitFixed, &one, NULL);
    newInitializedGPUVector(&y, "vector y", 1000, matrixInitFixed, &zero, NULL);



    EPARSE_CHECK_RETURN(prodMatrixVector(A, false, x, y))
 EPARSE_CHECK_RETURN(cloneVector(&y_host, memoryCPU, y, NULL))

    float sum = 0.0;
    for(int i = 0;i < y_host->nrow;i++){
        sum += (y_host->data)[i];
    }

	check( sum == 1000000., "L1 %f error",sum);

    // Dot product 2
    newInitializedGPUMatrix(&A, "matrix A", 1000, 1000, matrixInitFixed, &two, NULL);
    newInitializedGPUVector(&x, "vector x", 1000, matrixInitFixed, &one, NULL);
    newInitializedGPUVector(&y, "vector y", 1000, matrixInitFixed, &zero, NULL);
    newInitializedGPUVector(&ones, "ones", 1000, matrixInitFixed, &one, NULL);


    EPARSE_CHECK_RETURN(prodMatrixVector(A, false, x, y))

    EPARSE_CHECK_RETURN(dot(y, ones, &sum))

      check( sum == 2000000, "L1 %f error",sum);

	exit(EXIT_SUCCESS);

error:
	exit(EXIT_FAILURE);


}


int main() {
    testMatrixVectorProduct();
}
