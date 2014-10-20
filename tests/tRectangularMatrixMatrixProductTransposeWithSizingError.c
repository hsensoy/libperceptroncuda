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


void testRectangularMatrixMatrixProductTransposeWithSizingError(){

    // Dot product 4
    newInitializedCPUMatrix(&A, "matrix A", 1000, 100, matrixInitFixed, &one, NULL);
    newInitializedCPUMatrix(&B, "matrix B", 1000, 100, matrixInitFixed, &one, NULL);
    newInitializedCPUMatrix(&C, "matrix C", 10000, 10000, matrixInitFixed, &zero, NULL);


    CU_ASSERT_EQUAL(eparseColumnNumberMissmatch, prodMatrixMatrix(A,B, true, C))

    newInitializedCPUVector(&y, "vector y", 100, matrixInitFixed, &zero, NULL);

    CU_ASSERT_EQUAL(eparseColumnNumberMissmatch, prodMatrixVector(C, false, ones, y))

    float sum;

    CU_ASSERT_EQUAL(eparseColumnNumberMissmatch, dot(y, ones, &sum))

}


int main() {
    testRectangularMatrixMatrixProductTransposeWithSizingError();
}
