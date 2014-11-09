#include <stdio.h>
#include <stdlib.h>
#include "perceptron.h"
#include "debug.h"


void succesiveUpdatandScore() {
   Perceptron_t pkp = newPolynomialKernelPerceptron(2, 1.);


    Vector_t v = NULL;

    float somevalue = 1.;
    int nupdate = 40;
    newInitializedCPUVector(&v, "vector", 180, matrixInitFixed, &somevalue, NULL);

	Progress_t ptested = NULL;
    EPARSE_CHECK_RETURN(newProgress(&ptested, "dependency arc score", 40000 * 400, 0.01))

	long hvidx = 0;
    for (int i = 0; i < nupdate; i++) {

        float result;

		for (int _ai = 0; _ai < 400 ; _ai++){
        	EPARSE_CHECK_RETURN(score(pkp,v,false,&result))
        	

        	bool istick = tickProgress(ptested);
        
        	if (istick)
        		printf("Score %d: %f\n", i, result);
        }


        EPARSE_CHECK_RETURN(update(pkp,v,hvidx++,1))
        EPARSE_CHECK_RETURN(update(pkp,v,hvidx++,1))
        EPARSE_CHECK_RETURN(update(pkp,v,hvidx++,-1))
        EPARSE_CHECK_RETURN(update(pkp,v,hvidx++,-1))

        check(((KernelPerceptron_t)pkp->pDeriveObj)->kernel->matrix->ncol == hvidx, "Expected number of sv %ld violates the truth %ld",((KernelPerceptron_t)pkp->pDeriveObj)->kernel->matrix->ncol, hvidx )

    }

    EPARSE_CHECK_RETURN(deletePerceptron(pkp))

    return;

    error:
        exit(EXIT_FAILURE);
}

int main() {
    succesiveUpdatandScore();
}
