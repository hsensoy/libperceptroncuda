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
    check(1000 == result, "Result error %f in BLAS L1", result);

    newInitializedGPUVector(&a, "vector A", 1000, matrixInitFixed, &one, NULL);
    newInitializedGPUVector(&b, "vector B", 1000, matrixInitFixed, &two, NULL);

    dot(a, b, &result);

    check(2000 == result, "Result error %f in BLAS L1", result);

    newInitializedGPUVector(&a, "vector A", 1000, matrixInitFixed, &three, NULL);
    newInitializedGPUVector(&b, "vector B", 1000, matrixInitFixed, &three, NULL);

    EPARSE_CHECK_RETURN(dot(a, b, &result))
    
    check(9000 == result, "Result error %f in BLAS L1", result);

	exit(EXIT_SUCCESS);
error:
	exit(EXIT_FAILURE);

}

int main() {
    testDotProduct();
}
