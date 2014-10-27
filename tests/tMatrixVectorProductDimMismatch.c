#include <stdio.h>
#include <stdlib.h>
#include "epblas/epblas.h"

/*
 * CUnit Test Suite
 */

Matrix_t A;
Matrix_t x;
Vector_t y;
Vector_t ones;

float zero=0.,one=1.,two=2.,three,result;


void testMatrixVectorProductDimMismatch(){
    // Dot product 3
    newInitializedGPUMatrix(&A, "matrix A", 100, 1000, matrixInitFixed, &two, NULL);
    newInitializedGPUVector(&x, "vector x", 1000, matrixInitFixed, &one, NULL);
    newInitializedGPUVector(&y, "vector y", 1000, matrixInitFixed, &zero, NULL);


    check(eparseColumnNumberMissmatch == prodMatrixVector(A, true, x, y), "Matrix vector product could not capture dimension mismatch");

	exit(EXIT_SUCCESS);

error:
	exit(EXIT_FAILURE);
}


int main() {
    testMatrixVectorProductDimMismatch();
}
