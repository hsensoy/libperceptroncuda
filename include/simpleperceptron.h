#ifndef SIMPLE_PERCEPTRON_H


#include <stddef.h>
#include "epblas/epblas.h"
#include "epblas/eputil.h"
#include "debug.h"


struct SimplePerceptron_st{

};

typedef struct SimplePerceptron_st* SimplePerceptron_t;



SimplePerceptron_t __newSimplePerceptron();
eparseError_t deleteSimplePerceptron(SimplePerceptron_t p);


eparseError_t scoreSimplePerceptron(SimplePerceptron_t kp, Vector_t inst, bool avg, float *s);
eparseError_t scoreBatchSimplePerceptron(SimplePerceptron_t kp, Matrix_t instarr, bool avg, Vector_t *result);
eparseError_t updateSimplePerceptron(SimplePerceptron_t pkp, Vector_t sv, long svidx, float change);
eparseError_t recomputeSimplePerceptronAvgWeight(SimplePerceptron_t p);

eparseError_t dumpSimplePerceptron(FILE *fp, SimplePerceptron_t kp);
eparseError_t loadSimplePerceptron(FILE *fp, void **kp);

eparseError_t snapshotBestSimplePerceptron(SimplePerceptron_t sp);


#endif


