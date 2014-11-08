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


void testRectangularMatrixMatrixProductTranspose(){

    // Dot product 4
    newInitializedGPUMatrix(&A, "matrix A", 100, 1000, matrixInitFixed, &one, NULL);
    newInitializedGPUMatrix(&B, "matrix B", 100, 1000, matrixInitFixed, &one, NULL);
    newInitializedGPUMatrix(&C, "matrix C", 1000, 1000, matrixInitFixed, &zero, NULL);


    EPARSE_CHECK_RETURN(prodMatrixMatrix(A,true, B, C))

    newInitializedGPUVector(&y, "vector y", 1000, matrixInitFixed, &zero, NULL);
    newInitializedGPUVector(&ones, "vector 1s", 1000, matrixInitFixed, &one, NULL);

    EPARSE_CHECK_RETURN(prodMatrixVector(C, false, ones, y))

    float sum;
    EPARSE_CHECK_RETURN(dot(y, ones, &sum))

    check (100000000 == sum,"Dot product is %f",sum);

	exit(EXIT_SUCCESS);
error:
	exit(EXIT_FAILURE);
}


int main() {
    testRectangularMatrixMatrixProductTranspose();
}
