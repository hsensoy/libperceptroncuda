//
// Created by husnu sensoy on 11/04/15.
//

#ifndef PERCEPTRONMKL_FEATURETRANSFORM_H
#define PERCEPTRONMKL_FEATURETRANSFORM_H


#include "epblas/epblas.h"
#include <math.h>

#define TRANSFORM_BATCH_SIZE 5000


enum FeatureTransform {
    KERNAPROX_NONE,
    KERNAPROX_RBF_SAMPLER,
    /**
 * todo: We rather use kernel perceptron for the time being.
 */
            KERNAPROX_EXACT_POLY,

    /**
     * todo: Nystroem will be implemented in further releases.
     */
            KERNAPROX_NYSTROEM
};


struct FeatureTransformer_st {
    enum FeatureTransform type;

    void *pDeriveObj;


};

typedef struct FeatureTransformer_st *FeatureTransformer_t;


struct RBFSampler_st {
    Matrix_t samples;
    float sigma;

    long nsample;                 // number of iid gaussian omega vectors to be sampled
    long d;          // input dimensionality
    float scaler;   // sqrt(1/nsample)


    Vector_t partial_inst;
    
    Matrix_t in_cache;
    Matrix_t partial_matrix;
    Matrix_t out_dev;

};

typedef struct RBFSampler_st *RBFSampler_t;


/**
 * Generates of 2D dimensional vectors
 */
FeatureTransformer_t newRBFSampler(long D, float sigma);


eparseError_t deleteFeatureTransformer(FeatureTransformer_t ft);

/**
 *
 */
eparseError_t transform(FeatureTransformer_t ft, Vector_t in, Vector_t *out);


eparseError_t transformBatch(FeatureTransformer_t ft, Matrix_t in, Matrix_t *out);


#endif //PERCEPTRONMKL_FEATURETRANSFORM_H
