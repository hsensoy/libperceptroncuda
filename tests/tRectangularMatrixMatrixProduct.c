#include <stdio.h>
#include <stdlib.h>
#include "epblas/epblas.h"

/*
 * CUnit Test Suite
 */

Matrix_t A;
Matrix_t B;
Matrix_t C;
Matrix_t x;
Vector_t y;
Vector_t ones;

float zero=0.,one=1.,two,three,result;


void testRectangularMatrixMatrixProduct(){

    // Dot product 4
    newInitializedGPUMatrix(&A, "matrix A", 1000, 100, matrixInitFixed, &one, NULL);
    newInitializedGPUMatrix(&B, "matrix B", 100, 1000, matrixInitFixed, &one, NULL);
    newInitializedGPUMatrix(&C, "matrix C", 1000, 1000, matrixInitFixed, &zero, NULL);


    EPARSE_CHECK_RETURN(prodMatrixMatrix(A,B, false, C))

    newInitializedGPUVector(&y, "vector y", 1000, matrixInitFixed, &zero, NULL);
    newInitializedGPUVector(&ones, "vector 1s", 1000, matrixInitFixed, &one, NULL);
    EPARSE_CHECK_RETURN(prodMatrixVector(C, false, ones, y))

    float sum;
    EPARSE_CHECK_RETURN(dot(y, ones, &sum))

    check(100000000 == sum,"Dot product result is %f",sum);

	exit(EXIT_SUCCESS);
error:
	exit(EXIT_FAILURE);
}


int main() {
    testRectangularMatrixMatrixProduct();
}
