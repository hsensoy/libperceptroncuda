#include <stdio.h>
#include <stdlib.h>
#include "perceptron.h"
#include "debug.h"


void succesiveUpdatandScore() {
    Perceptron_t pkp = newPolynomialKernelPerceptron(2, 1.);


    Vector_t v = NULL;

    float somevalue = 1.;
    newInitializedCPUVector(&v, "vector", 20, matrixInitFixed, &somevalue, NULL);

    log_info("Allocation is done");

    for (int i = 0; i < 20; i++) {

        float result;

        EPARSE_CHECK_RETURN(score(pkp,v,false,&result))

        printf("Score %d: %f\n", i, result);


        EPARSE_CHECK_RETURN(update(pkp,v,i,1))

        check(((KernelPerceptron_t)pkp->pDeriveObj)->kernel->matrix->nrow == (i+1), "Expected number of sv %ld violates the truth %d",((KernelPerceptron_t)pkp->pDeriveObj)->kernel->matrix->nrow, i )

    }

    EPARSE_CHECK_RETURN(deletePerceptron(pkp))

    return;

    error:
    exit(EXIT_FAILURE);
}

int main() {
    succesiveUpdatandScore();
}
