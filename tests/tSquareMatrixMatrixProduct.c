#include <stdio.h>
#include <stdlib.h>
#include "epblas/epblas.h"

/*
 * CUnit Test Suite
 */

Matrix_t A=NULL;
Matrix_t B=NULL;
Matrix_t C=NULL;
Matrix_t x=NULL;
Vector_t y=NULL;
Vector_t ones=NULL;

float zero=0.,one=1.,two,three,result;


void testSquareMatrixMatrixProduct(){

    // Dot product 4
    newInitializedGPUMatrix(&A, "matrix A", 1000, 1000, matrixInitFixed, &one, NULL);
    newInitializedGPUMatrix(&B, "matrix B", 1000, 1000, matrixInitFixed, &one, NULL);
    newInitializedGPUMatrix(&C, "matrix C", 1000, 1000, matrixInitFixed, &zero, NULL);


    EPARSE_CHECK_RETURN(prodMatrixMatrix(A,B, false, C))

    newInitializedGPUVector(&y, "vector y", 1000, matrixInitFixed, &zero, NULL);
    newInitializedGPUVector(&ones, "vector ones", 1000, matrixInitFixed, &one, NULL);
    EPARSE_CHECK_RETURN(prodMatrixVector(C, false, ones, y))

    float sum;
    EPARSE_CHECK_RETURN(dot(y, ones, &sum))

    check(1000000000 == sum,"BLAS L1 Result Error %f", sum);

	exit(EXIT_SUCCESS);

error:
	exit(EXIT_FAILURE);
}


int main() {
    testSquareMatrixMatrixProduct();
}
