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


void testMatrixVectorProductDimMismatch(){
    // Dot product 3
    newInitializedCPUMatrix(&A, "matrix A", 100, 1000, matrixInitFixed, &two, NULL);
    newInitializedCPUVector(&x, "vector x", 1000, matrixInitFixed, &one, NULL);
    newInitializedCPUVector(&y, "vector y", 1000, matrixInitFixed, &zero, NULL);


    CU_ASSERT_EQUAL(eparseColumnNumberMissmatch,prodMatrixVector(A, true, x, y))
}


int main() {
    testMatrixVectorProductDimMismatch();
}
