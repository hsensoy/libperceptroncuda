#include <stdio.h>
#include <stdlib.h>
#include <CUnit/Basic.h>
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

float zero,one,two,three,result;


void testRectangularMatrixMatrixProduct(){

    // Dot product 4
    newInitializedCPUMatrix(&A, "matrix A", 1000, 100, matrixInitFixed, &one, NULL);
    newInitializedCPUMatrix(&B, "matrix B", 100, 1000, matrixInitFixed, &one, NULL);
    newInitializedCPUMatrix(&C, "matrix C", 1000, 1000, matrixInitFixed, &zero, NULL);


    EPARSE_CHECK_RETURN(prodMatrixMatrix(A,B, false, C))

    newInitializedCPUVector(&y, "vector y", 1000, matrixInitFixed, &zero, NULL);
    EPARSE_CHECK_RETURN(prodMatrixVector(C, false, ones, y))

    float sum;
    EPARSE_CHECK_RETURN(dot(y, ones, &sum))

    CU_ASSERT_EQUAL(100000000, sum);
}


int main() {
    testRectangularMatrixMatrixProduct();
}
