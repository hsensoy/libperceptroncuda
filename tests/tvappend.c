#include <stdio.h>
#include <stdlib.h>
#include "epblas/epblas.h"

#define N 10000000

Vector_t v2N = NULL;
Vector_t v1N_1 = NULL;
Vector_t v1N_2 = NULL;

float one=1.;

void testAppendOneByOne(){
    newInitializedCPUVector(&v1N_1, "vector 100-1", N, matrixInitFixed, &one, NULL);
    
    for(size_t i = 0; i < N; ++i)
    {
        EPARSE_CHECK_RETURN(vappend(&v1N_2, memoryCPU, "vector100-2",one));
    }
    
    check(vequal(v1N_1, v1N_2), "%s != %s",v1N_1->identifier, v1N_2->identifier);
    
    deleteVector(v1N_1);
    deleteVector(v1N_2);
    
    return;
    
    error:
    exit(EXIT_FAILURE);
}

void testAppendOneByOne_2(){
    newInitializedGPUVector(&v1N_1, "vector 100-1", N, matrixInitFixed, &one, NULL);

    for(size_t i = 0; i < N; ++i)
    {
        EPARSE_CHECK_RETURN(vappend(&v1N_2, memoryGPU, "vector100-2",one));
    }

    check(vequal(v1N_1, v1N_2), "%s != %s",v1N_1->identifier, v1N_2->identifier);

    deleteVector(v1N_1);
    deleteVector(v1N_2);

    return;

    error:
    exit(EXIT_FAILURE);
}

void testNMore(){
    v2N = NULL;
    v1N_1 = NULL;
    newInitializedCPUVector(&v2N, "vector 200", 2 * N, matrixInitFixed, &one, NULL);
    newInitializedCPUVector(&v1N_1, "vector 100-1", N, matrixInitFixed, &one, NULL);
    
    for(size_t i = 0; i < N; ++i)
    {
        EPARSE_CHECK_RETURN(vappend(&v1N_1, memoryCPU, "vector 100-1",one));
    }
    
    check(vequal(v1N_1, v2N), "%s != %s",v1N_1->identifier, v2N->identifier);
    
    deleteVector(v1N_1);
    deleteVector(v2N);
    
    return;
    
    error:
    exit(EXIT_FAILURE);
}

void testNMore_2(){
    v2N = NULL;
    v1N_1 = NULL;
    newInitializedGPUVector(&v2N, "vector 200", 2 * N, matrixInitFixed, &one, NULL);
    newInitializedGPUVector(&v1N_1, "vector 100-1", N, matrixInitFixed, &one, NULL);

    for(size_t i = 0; i < N; ++i)
    {
        EPARSE_CHECK_RETURN(vappend(&v1N_1, memoryGPU, "vector 100-1",one));
    }

    check(vequal(v1N_1, v2N), "%s != %s",v1N_1->identifier, v2N->identifier);

    deleteVector(v1N_1);
    deleteVector(v2N);

    return;

    error:
    exit(EXIT_FAILURE);
}

void testNMoreBatchVector(){
    v2N = NULL;
    v1N_1 = NULL;
    v1N_2 = NULL;
    newInitializedCPUVector(&v2N, "vector 200", 2*N, matrixInitFixed, &one, NULL);
    newInitializedCPUVector(&v1N_1, "vector 100-1", N, matrixInitFixed, &one, NULL);
    
    EPARSE_CHECK_RETURN(vappend_vector(&v1N_2, memoryCPU,"vector 100-2",v1N_1))
    EPARSE_CHECK_RETURN(vappend_vector(&v1N_2, memoryCPU,"vector 100-2",v1N_1))
    

    check(vequal(v1N_2, v2N), "%s != %s",v1N_2->identifier, v2N->identifier);
    
    deleteVector(v1N_1);
    deleteVector(v1N_2);
    deleteVector(v2N);
    
    return;
    
    error:
    exit(EXIT_FAILURE);
}

void testNMoreBatchVector_2(){
    v2N = NULL;
    v1N_1 = NULL;
    v1N_2 = NULL;
    newInitializedCPUVector(&v2N, "vector 200", 2*N, matrixInitFixed, &one, NULL);
    newInitializedCPUVector(&v1N_1, "vector 100-1", N, matrixInitFixed, &one, NULL);

    EPARSE_CHECK_RETURN(vappend_vector(&v1N_2, memoryGPU,"vector 100-2",v1N_1))
    EPARSE_CHECK_RETURN(vappend_vector(&v1N_2, memoryGPU,"vector 100-2",v1N_1))


    check(vequal(v1N_2, v2N), "%s != %s",v1N_2->identifier, v2N->identifier);

    deleteVector(v1N_1);
    deleteVector(v1N_2);
    deleteVector(v2N);

    return;

    error:
    exit(EXIT_FAILURE);
}

void testNMoreBatchArray(){
    v2N = NULL;
    v1N_2 = NULL;
    
    float *arr = (float*)malloc(sizeof(float) * N);
    check(arr != NULL,"Memory allocation problem");
    
    for(size_t i = 0; i < N; ++i)
    {
        arr[i] = one;
    }
    
    
    newInitializedCPUVector(&v2N, "vector 200", 2*N, matrixInitFixed, &one, NULL);
    
    EPARSE_CHECK_RETURN(vappend_array(&v1N_2, memoryCPU,"vector 100-2",N, arr))
    EPARSE_CHECK_RETURN(vappend_array(&v1N_2, memoryCPU,"vector 100-2",N, arr))
    

    check(vequal(v1N_2, v2N), "%s != %s",v1N_2->identifier, v2N->identifier);
    
    deleteVector(v1N_2);
    deleteVector(v2N);
    
    return;
    
    error:
    exit(EXIT_FAILURE);
}

void testNMoreBatchArray_2(){
    v2N = NULL;
    v1N_2 = NULL;

    float *arr = (float*)malloc(sizeof(float) * N);
    check(arr != NULL,"Memory allocation problem");

    for(size_t i = 0; i < N; ++i)
    {
        arr[i] = one;
    }


    newInitializedCPUVector(&v2N, "vector 200", 2*N, matrixInitFixed, &one, NULL);

    EPARSE_CHECK_RETURN(vappend_array(&v1N_2, memoryCPU,"vector 100-2",N, arr))
    EPARSE_CHECK_RETURN(vappend_array(&v1N_2, memoryCPU,"vector 100-2",N, arr))


    check(vequal(v1N_2, v2N), "%s != %s",v1N_2->identifier, v2N->identifier);

    deleteVector(v1N_2);
    deleteVector(v2N);

    return;

    error:
    exit(EXIT_FAILURE);
}

int main() {
    testAppendOneByOne();
    testNMore();
    testNMoreBatchVector();
    testNMoreBatchArray();

    testAppendOneByOne_2();
    testNMore_2();
    testNMoreBatchVector_2();
    testNMoreBatchArray_2();
}
