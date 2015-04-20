#include <stdio.h>
#include <stdlib.h>
#include "perceptron.h"
#include "featuretransform.h"
#include "debug.h"
#include <mkl.h>

#define NSENTENCE 10000
#define AVG_SENTENCE_LENGTH 20
#define XFORMED_EMBEDDING_LENGTH 365


void succesiveUpdatandScore() {
    Perceptron_t pkp = newSimplePerceptron();



    FeatureTransformer_t  ft = newRBFSampler( 80000, 10.f);

    Vector_t v = NULL;
    float somevalue = 1.;

    newInitializedCPUVector(&v, "vector", XFORMED_EMBEDDING_LENGTH, matrixInitNone, NULL, NULL);

    for (long k = 0; k < XFORMED_EMBEDDING_LENGTH; ++k) {
        (v->data)[k] = 1;
    }

    printMatrix("raw input",v,stdout);

    Matrix_t vBatch = NULL;

    for (int l = 0; l < AVG_SENTENCE_LENGTH*AVG_SENTENCE_LENGTH; ++l) {
        EPARSE_CHECK_RETURN(hstack(&vBatch,memoryCPU,"Batch Input",v,false,false))
    }


    printMatrix("Batch Input", vBatch,stdout);

    Vector_t v_nl = NULL;
    EPARSE_CHECK_RETURN(transform(ft,v,&v_nl))

    printMatrix( "Gaussian Samples for RBF Sampler", ((RBFSampler_t)ft->pDeriveObj)->samples,stdout);

    printMatrix("transformed",v_nl,stdout);


    Matrix_t partial_m = NULL;
    float zero = 0.f;
    newInitializedMatrix(&(partial_m),memoryCPU,"Partial Matrix",((RBFSampler_t)ft->pDeriveObj)->nsample, vBatch->ncol, matrixInitFixed,&zero,NULL);
    EPARSE_CHECK_RETURN(prodMatrixMatrix(((RBFSampler_t)ft->pDeriveObj)->samples,false,vBatch , partial_m));

    printMatrix("partial transformed",partial_m,stdout);

    Matrix_t vBatch_nl = NULL;
    newInitializedMatrix(&(vBatch_nl),memoryCPU,"Partial Matrix",((RBFSampler_t)ft->pDeriveObj)->nsample*2, vBatch->ncol , matrixInitFixed,&zero,NULL);

    EPARSE_CHECK_RETURN(CosSinMatrix(partial_m,vBatch_nl))

    printMatrix("trigo-transformed",vBatch_nl,stdout);


    cblas_sscal(vBatch_nl->n,((RBFSampler_t)ft->pDeriveObj)->scaler ,vBatch_nl->data, 1);

    printMatrix("transformed",vBatch_nl,stdout);

    EPARSE_CHECK_RETURN(transformBatch(ft,vBatch,&vBatch_nl))

    printMatrix("transformed",vBatch_nl,stdout);




    Vector_t vScore = NULL;






    log_info("Allocation is done");
    Progress_t ptested = NULL;
    CHECK_RETURN(newProgress(&ptested, "test sentences", NSENTENCE, 0.1))



    EPARSE_CHECK_RETURN(transformBatch(ft,vBatch,&vBatch_nl))

    int hvidx = 0;
    for (int i = 0; i < NSENTENCE; i++) {

        float result;

        debug("***** %d. 400 arc pack is stacked *****", i + 1);

        ///EPARSE_CHECK_RETURN(score(pkp,v,false,&result))



        //EPARSE_CHECK_RETURN(transform(ft,v,&v_nl))


        EPARSE_CHECK_RETURN(scoreBatch(pkp, vBatch_nl, false, &vScore))


        for (int _i = 0; _i < 2; _i++) {
            EPARSE_CHECK_RETURN(update(pkp,v_nl,hvidx++,1))
            EPARSE_CHECK_RETURN(update(pkp,v_nl,hvidx++,-1))
        }

        bool istick = tickProgress(ptested);

    }

    EPARSE_CHECK_RETURN(deletePerceptron(pkp))

    return;
    error:
        exit(EXIT_FAILURE);

}

int main() {
    succesiveUpdatandScore();
}
