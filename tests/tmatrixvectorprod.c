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


void testMatrixVectorProduct() {

    // Dot product 1
    newInitializedCPUMatrix(&A, "matrix A", 1000, 1000, matrixInitFixed, &one, NULL);
    newInitializedCPUVector(&x, "vector x", 1000, matrixInitFixed, &one, NULL);
    newInitializedCPUVector(&y, "vector y", 1000, matrixInitFixed, &zero, NULL);



    EPARSE_CHECK_RETURN(prodMatrixVector(A, false, x, y))

    float sum = 0.0;
    for(int i = 0;i < y->nrow;i++){
        sum += (y->data)[i];
    }

    CU_ASSERT_EQUAL(1000000, sum);

    // Dot product 2
    newInitializedCPUMatrix(&A, "matrix A", 1000, 1000, matrixInitFixed, &two, NULL);
    newInitializedCPUVector(&x, "vector x", 1000, matrixInitFixed, &one, NULL);
    newInitializedCPUVector(&y, "vector y", 1000, matrixInitFixed, &zero, NULL);


    EPARSE_CHECK_RETURN(prodMatrixVector(A, false, x, y))

    EPARSE_CHECK_RETURN(dot(y, ones, &sum))


    CU_ASSERT_EQUAL(2000000, sum);


}


int main() {
    testMatrixVectorProduct();
}
