#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "epblas/epblas.h"
#include "epblas/eputil.h"
#include "debug.h"
#include "perceptron_common.h"



struct Perceptron_st{
    enum PerceptronType type;

    void *pDeriveObj;
};

typedef struct Perceptron_st* Perceptron_t;

Perceptron_t newPolynomialKernelPerceptron(int power, float bias);
Perceptron_t newSimplePerceptron();

eparseError_t deletePerceptron(Perceptron_t p);


eparseError_t score(Perceptron_t kp, Vector_t inst, bool avg, float *s);
eparseError_t scoreBatch(Perceptron_t kp, Matrix_t instarr, bool avg, Vector_t *result);
eparseError_t update(Perceptron_t pkp, Vector_t sv, long svidx, float change);
eparseError_t recomputeAvgWeight(Perceptron_t p);

eparseError_t snapshotBest(Perceptron_t p);

eparseError_t dumpPerceptronModel(FILE *fp, Perceptron_t kp);
eparseError_t loadPerceptronModel(FILE *fp, Perceptron_t *kp);

eparseError_t showStats(Perceptron_t p);

/**
* Parallel execution control functions.
*/
eparseError_t setPerceptronParallism(int nslave);
eparseError_t getPerceptronMaxParallism(int *nslave);
eparseError_t getPerceptronDynamicParallism(bool *dynamic);



#endif
