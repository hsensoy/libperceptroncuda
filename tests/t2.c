#include <stdio.h>
#include <stdlib.h>
#include "perceptron.h"
#include "debug.h"
#include <cuda_profiler_api.h>

#define NSENTENCE 10000
#define AVG_SENTENCE_LENGTH 20
#define XFORMED_EMBEDDING_LENGTH 720


void succesiveUpdatandScore() {
    Perceptron_t pkp = newPolynomialKernelPerceptron(2, 1.);


    Vector_t v = NULL,vScore= NULL;
    Matrix_t vBatch = NULL;

    float somevalue = 1.;
    
    newInitializedGPUVector(&v, "vector", XFORMED_EMBEDDING_LENGTH, matrixInitFixed, &somevalue, NULL);


    log_info("Allocation is done");
	Progress_t ptested = NULL;
	EPARSE_CHECK_RETURN(newProgress(&ptested, "test sentences", NSENTENCE, 0.1))
	
	EPARSE_CHECK_RETURN(newInitializedGPUMatrix(&vBatch, "arc matrix", AVG_SENTENCE_LENGTH * AVG_SENTENCE_LENGTH, XFORMED_EMBEDDING_LENGTH, matrixInitNone, NULL,NULL))
	
    for (int j = 0; j < AVG_SENTENCE_LENGTH * AVG_SENTENCE_LENGTH; j++)
		EPARSE_CHECK_RETURN(matrixDatacpyAnyToAny(vBatch, j * XFORMED_EMBEDDING_LENGTH, v, 0, sizeof(float) * v->n))
	
	int hvidx = 0;
    for (int i = 0; i < NSENTENCE; i++) {

        float result;
        
		debug("***** %d. 400 arc pack is stacked *****", i+1);
        	
        ///EPARSE_CHECK_RETURN(score(pkp,v,false,&result))
        
        //log_info("Sentence %d", i);
        EPARSE_CHECK_RETURN(scoreBatch(pkp, vBatch, false, &vScore))
		
		for (int _i = 0; _i < 2; _i++) {
        	EPARSE_CHECK_RETURN(update(pkp,v,hvidx++,1))
        	EPARSE_CHECK_RETURN(update(pkp,v,hvidx++,-1))
        }
        
        bool istick = tickProgress(ptested);
        
        if (istick)
        	log_info("%ld hypothesis vectors",((KernelPerceptron_t)pkp->pDeriveObj)->kernel->matrix->nrow);

        check(((KernelPerceptron_t)pkp->pDeriveObj)->kernel->matrix->nrow == hvidx, "Expected number of sv %ld violates the truth %d",((KernelPerceptron_t)pkp->pDeriveObj)->kernel->matrix->nrow, i )

    }

    EPARSE_CHECK_RETURN(deletePerceptron(pkp))

    return;
    
    

    error:
    exit(EXIT_FAILURE);
}

int main() {
    succesiveUpdatandScore();
}
