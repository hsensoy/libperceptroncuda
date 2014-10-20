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


void testMatrixVectorProductwithTranspose(){

    // Dot product 4
    newInitializedCPUMatrix(&A, "matrix A", 100, 1000, matrixInitFixed, &two, NULL);
    newInitializedCPUVector(&x, "vector x", 100, matrixInitFixed, &one, NULL);
    newInitializedCPUVector(&y, "vector y", 1000, matrixInitFixed, &zero, NULL);

    float sum;
    EPARSE_CHECK_RETURN(prodMatrixVector(A, true, x, y))

    EPARSE_CHECK_RETURN(dot(y, ones, &sum))


    CU_ASSERT_EQUAL(200000, sum);
}


int main() {
    testMatrixVectorProductwithTranspose();
}
