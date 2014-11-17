#include <stdio.h>
#include <stdlib.h>
#include "perceptron.h"
#include "debug.h"
#include <cuda_profiler_api.h>
#include <math.h>

#define NSENTENCE 1000
#define AVG_SENTENCE_LENGTH 50
#define XFORMED_EMBEDDING_LENGTH 1000

#define EQUAL_WITH_EPSILON(expected, actual, epsilon) ( fabs ( (expected) - (actual)) <= epsilon )


void succesiveUpdatandScore() {
    Perceptron_t pkp = newPolynomialKernelPerceptron(2, 1.f);

	/*
		TODO Allocate deallocate succesively.
	*/
    Vector_t v = NULL;

    float somevalue = .1f;
    
    newInitializedCPUVector(&v, "vector", XFORMED_EMBEDDING_LENGTH, matrixInitFixed, &somevalue, NULL);

    log_info("Allocation is done");
	Progress_t ptested = NULL;
	EPARSE_CHECK_RETURN(newProgress(&ptested, "test sentences", NSENTENCE, 0.1))
		
	int hvidx = 0;
    for (int i = 0; i < NSENTENCE; i++) {    	
    	Vector_t vScore= NULL;
   	Matrix_t all = NULL;
    
	for (int _from = 0; _from <= AVG_SENTENCE_LENGTH; _from++) {
		for (int _to = 1; _to <= AVG_SENTENCE_LENGTH; _to++)
			if (_to != _from) {

                   			EPARSE_CHECK_RETURN(hstack(&all, memoryCPU, "all embeddings", v, false, false))
               		}
    	}
        ///EPARSE_CHECK_RETURN(score(pkp,v,false,&result))
        
        //log_info("Sentence %d", i);
        EPARSE_CHECK_RETURN(scoreBatch(pkp, all, false, &vScore))
		
		for (int _i = 0; _i < 2; _i++)
        	EPARSE_CHECK_RETURN(update(pkp,v,hvidx++, (i%2 == 0)?1.f:-1.f))
				
				/*
		for (int _i = 0; _i < 2; _i++) 
        	EPARSE_CHECK_RETURN(update(pkp,v,hvidx++,-1))
				*/
        
        bool istick = tickProgress(ptested);
        
        if (istick)
        	log_info("%ld hypothesis vectors",((KernelPerceptron_t)pkp->pDeriveObj)->kernel->matrix->nrow);

        check(((KernelPerceptron_t)pkp->pDeriveObj)->kernel->matrix->ncol == hvidx, "Expected number of sv %ld violates the truth %d",((KernelPerceptron_t)pkp->pDeriveObj)->kernel->matrix->ncol, i )
			
		
		long realidx = 0;
		log_info("Sentence %d",i);
	    for (int _from = 0; _from <= AVG_SENTENCE_LENGTH; _from++) {
	           for (int _to = 1; _to <= AVG_SENTENCE_LENGTH; _to++)
	               if (_to != _from) {
					   
			check( EQUAL_WITH_EPSILON( (vScore->data)[realidx], (i%2) * 2 * 121.f,.01f),"Expected %f Actual %f",(i%2) * 2 * 121.f,(vScore->data)[realidx]  )
						   
					   realidx++;
					   
	               }
	    }

 	deleteMatrix(all);
        deleteVector(vScore);

    }

    EPARSE_CHECK_RETURN(deletePerceptron(pkp))

    return;
    
    

    error:
    exit(EXIT_FAILURE);
}

int main() {
    succesiveUpdatandScore();
}
