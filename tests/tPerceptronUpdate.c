#include <stdio.h>
#include <stdlib.h>
#include "perceptron.h"
#include "debug.h"

#define NSENTENCE 40
#define AVG_SENTENCE_LENGTH 20
#define XFORMED_EMBEDDING_LENGTH 720


void succesiveUpdatandScore() {
    Perceptron_t pkp = newSimplePerceptron(NULL);




    Vector_t v = NULL;
    Matrix_t vBatch = NULL;

    float somevalue = 1.;
    float vScore = 0.f;

    newInitializedCPUVector(&v, "vector", XFORMED_EMBEDDING_LENGTH, matrixInitFixed, &somevalue, NULL);


    log_info("Allocation is done");
    Progress_t ptested = NULL;
    CHECK_RETURN(newProgress(&ptested, "test sentences", NSENTENCE, 0.1))



    int hvidx = 0;
    for (int i = 0; i < NSENTENCE; i++) {

        float result;

        debug("***** %d. 400 arc pack is stacked *****", i + 1);

        ///EPARSE_CHECK_RETURN(score(pkp,v,false,&result))

        //log_info("Sentence %d", i);
        for (int j = 0; j < AVG_SENTENCE_LENGTH * AVG_SENTENCE_LENGTH; j++) {
            EPARSE_CHECK_RETURN(score(pkp, v, false, &vScore))

            check(vScore == 0, "vscore is expected to be 0.0 where found %f",vScore);
        }



        for (int _i = 0; _i < 2; _i++) {
            EPARSE_CHECK_RETURN(update(pkp,v,hvidx++,1))
            EPARSE_CHECK_RETURN(update(pkp,v,hvidx++,-1))
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
