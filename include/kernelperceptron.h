#ifndef KERNEL_PERCEPTRON_H


#include "epblas/epblas.h"
#include "epblas/eputil.h"
#include "debug.h"
#include "util.h"
#include "perceptron_common.h"

KernelPerceptron_t __newPolynomialKernelPerceptron(int power, float bias);
eparseError_t deleteKernelPerceptron(KernelPerceptron_t p);


eparseError_t scoreKernelPerceptron(KernelPerceptron_t kp, Vector_t inst, bool avg, float *s);
eparseError_t scoreBatchKernelPerceptron(KernelPerceptron_t kp, Matrix_t instarr, bool avg, Vector_t *result);
eparseError_t updateKernelPerceptron(KernelPerceptron_t pkp, Vector_t sv, long svidx, float change);
eparseError_t recomputeKernelPerceptronAvgWeight(KernelPerceptron_t kp);

eparseError_t dumpKernelPerceptron(FILE *fp, KernelPerceptron_t kp);
eparseError_t loadKernelPerceptron(FILE *fp, void **kp);

eparseError_t showStatsKernelPerceptron(KernelPerceptron_t kp);

eparseError_t snapshotBestKernelPerceptron(KernelPerceptron_t kp);


#endif
