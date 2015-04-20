#include <stdio.h>
#include <stdlib.h>
#include "featuretransform.h"
#include "perceptron.h"
#include "debug.h"

void creatndDrop() {
    log_info("Creating simple perceptron");
    Perceptron_t pkp = newSimplePerceptron();

    log_info("Creating and RBF Sampler");
    FeatureTransformer_t ft = newRBFSampler(100000, 1.);

    log_info("Deleting RBF transfomer");
    EPARSE_CHECK_RETURN(deleteFeatureTransformer(ft))

    log_info("Deleting simple perceptron");
    EPARSE_CHECK_RETURN(deletePerceptron(pkp))
}

int main() {
    creatndDrop();
}
