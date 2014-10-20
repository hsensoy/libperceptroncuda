#include "simpleperceptron.h"


eparseError_t deleteSimplePerceptron(SimplePerceptron_t sp) {
    free(sp);

    return eparseSucess;

}

SimplePerceptron_t __newSimplePerceptron() {
    // TODO: Implement
    return NULL;
}

eparseError_t scoreSimplePerceptron(SimplePerceptron_t kp, Vector_t inst, bool avg, float *s) {
    // TODO: Implement
    return eparseColumnNumberMissmatch;
}

eparseError_t scoreBatchSimplePerceptron(SimplePerceptron_t kp, Matrix_t instarr, bool avg, Vector_t *result) {
    // TODO: Implement
    return eparseColumnNumberMissmatch;
}

eparseError_t updateSimplePerceptron(SimplePerceptron_t pkp, Vector_t sv, long svidx, float change) {
    // TODO: Implement
    return eparseColumnNumberMissmatch;
}

eparseError_t dumpSimplePerceptron(FILE *fp, SimplePerceptron_t kp) {
    // TODO: Implement
    return eparseColumnNumberMissmatch;
}

eparseError_t loadSimplePerceptron(FILE *fp, void **kp) {
    // TODO: Implement
    return eparseColumnNumberMissmatch;
}

eparseError_t recomputeSimplePerceptronAvgWeight(SimplePerceptron_t p){
    return eparseColumnNumberMissmatch;
}

eparseError_t snapshotBestSimplePerceptron(SimplePerceptron_t sp){


    return eparseColumnNumberMissmatch;
}