#include <stdio.h>
#include <stdlib.h>
#include "perceptron.h"
#include "featuretransform.h"
#include "debug.h"

#define NSENTENCE 10000
#define AVG_SENTENCE_LENGTH 20
#define XFORMED_EMBEDDING_LENGTH 365


void succesiveUpdatandScore() {
    Perceptron_t pkp = newSimplePerceptron();



    FeatureTransformer_t  ft = newRBFSampler(10000, 1.);


    Vector_t v = NULL;
    Vector_t v_nl = NULL;
    Matrix_t vBatch = NULL;

    Matrix_t vBatch_nl = NULL;

    float somevalue = 1.;
    Vector_t vScore = NULL;

    newInitializedCPUVector(&v, "vector", XFORMED_EMBEDDING_LENGTH, matrixInitFixed, &somevalue, NULL);



    EPARSE_CHECK_RETURN(transform(ft,v,&v_nl))
    log_info("Allocation is done");
    Progress_t ptested = NULL;
    CHECK_RETURN(newProgress(&ptested, "test sentences", NSENTENCE, 0.1))

    EPARSE_CHECK_RETURN(newInitializedCPUMatrix(&vBatch, "arc matrix", AVG_SENTENCE_LENGTH * AVG_SENTENCE_LENGTH, XFORMED_EMBEDDING_LENGTH, matrixInitNone, NULL,NULL))


    for (int j = 0; j < AVG_SENTENCE_LENGTH * AVG_SENTENCE_LENGTH; j++)
        memcpy(vBatch->data +  j * v_nl->n, v->data, sizeof(float) * v->n);

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
