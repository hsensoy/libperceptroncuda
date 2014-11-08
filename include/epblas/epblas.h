#ifndef EPBLAS_H
#define	EPBLAS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "eputil.h"

char* version();

enum memoryAllocationDevice_t {
    memoryGPU, memoryCPU
};

typedef enum memoryAllocationDevice_t memoryAllocationDevice_t;

struct Matrix_st {
    char *identifier;
    long nrow;
    long ncol;
    long n;

    long capacity;

    memoryAllocationDevice_t dev;

    bool isvector;
    float *data;
};

typedef struct Matrix_st* Matrix_t;
typedef struct Matrix_st* Vector_t;

enum matrixInitializer_t {
    matrixInitRandom, matrixInitFixed, matrixInitCArray, matrixInitNone
};

typedef enum matrixInitializer_t matrixInitializer_t;

// TODO: ensureMatrixCapacity should be private
eparseError_t ensureMatrixCapacity(Matrix_t mptr, long nentry);

eparseError_t newMatrix(Matrix_t *mptr, memoryAllocationDevice_t device,
        const char *id, long nrow, long ncol);

#define newVector(vptr, device, id, n){	\
	newMatrix( (vptr), (device), (id), (n), (1));	\
	(*vptr)->isvector = true;	\
}

/**
create a new initialized matrix

TODO: stream was set to be of curandGenerator_t type for CUDA
*/
eparseError_t newInitializedMatrix(Matrix_t *mptr,
        memoryAllocationDevice_t device, const char *id, long nrow, long ncol,
        matrixInitializer_t strategy, float *fix_value,
        void *stream);


#define newInitializedVector(vptr, device, id, n,strategy,fix_value,stream){	\
		EPARSE_CHECK_RETURN(newInitializedMatrix( (vptr), (device), (id), (n),1, (strategy), (fix_value),(stream)))	\
		(*vptr)->isvector = true;	\
}

#define newInitializedCPUVector(vptr, id, n,strategy,fix_value,stream)  newInitializedVector( (vptr),  memoryCPU, (id),  (n),  (strategy), (fix_value),  (stream))
#define newInitializedGPUVector(vptr, id, n,strategy,fix_value,stream)  newInitializedVector( (vptr),  memoryGPU, (id),  (n),  (strategy), (fix_value),  (stream))

#define newCPUMatrix( mptr,  id,   nrow,  ncol) newMatrix( (mptr), memoryCPU, (id), (nrow),  (ncol))
#define newGPUMatrix( mptr,  id,   nrow,  ncol) newMatrix( (mptr), memoryGPU, (id), (nrow),  (ncol))

#define newInitializedCPUMatrix(mptr, id,  nrow,  ncol,  strategy, fix_value,  stream) newInitializedMatrix( (mptr),  memoryCPU, (id),  (nrow),  (ncol),  (strategy), (fix_value),  (stream))
#define newInitializedGPUMatrix(mptr, id,  nrow,  ncol,  strategy, fix_value,  stream) newInitializedMatrix( (mptr),  memoryGPU, (id),  (nrow),  (ncol),  (strategy), (fix_value),  (stream))

eparseError_t prodMatrixVector(Matrix_t A, bool tA, Vector_t x, Vector_t y);
eparseError_t prodMatrixMatrix(Matrix_t A, bool tA, Matrix_t B, Matrix_t C);
eparseError_t powerMatrix(Matrix_t x, int power, Matrix_t y);
eparseError_t dot(Vector_t x, Vector_t y, float *result);





/**
clone a given matrix
*/
eparseError_t cloneMatrix(Matrix_t *dst, memoryAllocationDevice_t device,
        const Matrix_t src, const char *new_id);

eparseError_t mtrxcolcpy(Matrix_t *dst, memoryAllocationDevice_t device,
        const Matrix_t src, const char *new_id, long offsetcol, long ncol);

#define cloneVector(dst, device, src, new_id ) cloneMatrix((dst),(device),(src),(new_id))

/**
deallocate a matrix
*/
eparseError_t __deleteMatrix(Matrix_t m);

#define deleteVector(v){					\
	EPARSE_CHECK_RETURN(__deleteMatrix(v))	\
	v = NULL;								\
}

#define deleteMatrix(m){					\
	EPARSE_CHECK_RETURN(__deleteMatrix(m))	\
	m = NULL;								\
}



/**
*	Stack m2 next to m1 sequence vertically (row wise).
*
* @param m1 Target matrix
* @param device Target device of m1
* @param id Identifier for m1 if it is NULL
* @param m2 Source matrix
* @param releaseM2 whether to release source matrix memory.
* @return
*/
eparseError_t vstackMatrix(Matrix_t *m1, memoryAllocationDevice_t device,
        const char* id, Matrix_t m2, bool transposeM2, bool releaseM2);

#define vstackVector vstackMatrix
		
eparseError_t hstack(Matrix_t *m1, memoryAllocationDevice_t device, const char* id, Matrix_t m2, bool transposeM2, bool releaseM2);		



eparseError_t vappend(Vector_t *v, memoryAllocationDevice_t device, const char* id, float value);


void printMatrix(const char* heading, Matrix_t m, FILE *fp);
char* humanreadable_size(size_t bytes);

void setParallism(int nslave);
bool getDynamicParallism();
int getMaxParallism();

// TODO: Implement for mkl based one also
eparseError_t matrixDatacpyAnyToAny(Matrix_t dest, long dest_offset,
        Matrix_t src, long src_offset, size_t bytes);

#endif
