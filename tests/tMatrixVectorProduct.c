#include <stdio.h>
#include <stdlib.h>
#include "epblas/epblas.h"

/*
 * CUnit Test Suite
 */

Matrix_t A=NULL;
Matrix_t x=NULL;
Vector_t y=NULL;
Vector_t ones=NULL;

float zero=0.,one=1.,two=2.,three,result;


void testMatrixVectorProductwithTranspose(){

    // Dot product 4
    newInitializedGPUMatrix(&A, "matrix A", 10000, 365, matrixInitFixed, &two, NULL);
    newInitializedGPUVector(&x, "vector x", 365, matrixInitFixed, &one, NULL);
    newInitializedGPUVector(&y, "vector y", 10000, matrixInitFixed, &zero, NULL);
    newInitializedGPUVector(&ones, "vector 1s", 10000, matrixInitFixed, &one, NULL);

    float sum;
    EPARSE_CHECK_RETURN(prodMatrixVector(A,false, x, y))

    EPARSE_CHECK_RETURN(dot(y, ones, &sum))


    check(7300000== sum,"Sum is %f",sum);

	exit(EXIT_SUCCESS);

error:
	exit(EXIT_FAILURE);
}


int main() {
    testMatrixVectorProductwithTranspose();
}
