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

void testDotProduct() {
    Vector_t a = NULL;
    Vector_t b = NULL;

    float one = 1;
    float two = 2;
    float three = 3;
    float result ;

    newInitializedGPUVector(&a, "vector A", 1000, matrixInitFixed, &one, NULL);
    newInitializedGPUVector(&b, "vector B", 1000, matrixInitFixed, &one, NULL);

    dot(a, b, &result);
    CU_ASSERT_EQUAL(1000, result);

    newInitializedCPUVector(&a, "vector A", 1000, matrixInitFixed, &one, NULL);
    newInitializedCPUVector(&b, "vector B", 1000, matrixInitFixed, &two, NULL);

    dot(a, b, &result);

    CU_ASSERT_EQUAL(2000, result);

    newInitializedCPUVector(&a, "vector A", 1000, matrixInitFixed, &three, NULL);
    newInitializedCPUVector(&b, "vector B", 1000, matrixInitFixed, &three, NULL);

    EPARSE_CHECK_RETURN(dot(a, b, &result))

    CU_ASSERT_EQUAL(9000, result);
}

int main() {
    testDotProduct();
}
