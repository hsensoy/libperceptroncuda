#ifndef PERCEPTRON_COMMON_H
#define  PERCEPTRON_COMMON_H
/**
*
* NONE: No budgeting at all
* RANDOMIZED: Choose one randomly out of hypothesis vector with alpha value equal to 1.
*/
enum BudgetMethod{
    NONE,
    RANDOMIZED
};

typedef enum BudgetMethod BudgetMethod;


enum PerceptronType{
    SIMPLE_PERCEPTRON,
    KERNEL_PERCEPTRON
};

typedef enum PerceptronType PerceptronType;


enum KernelType{
    POLYNOMIAL_KERNEL,
    RBF_KERNEL
};

typedef enum KernelType KernelType;


struct Kernel_st {
    Matrix_t matrix;

    Vector_t alpha;
    Vector_t alpha_avg;
    Vector_t beta;
};

typedef struct Kernel_st* Kernel_t;


struct KernelPerceptron_st{
    enum KernelType kerneltype;
    int c;
    int best_numit;

    Kernel_t kernel;
    Kernel_t best_kernel;

    void *pDerivedObj;


    // Object below are temporary structures used at intermediate steps of computation.
    Matrix_t t_instBatch;
    Matrix_t t_yBatch;
    Matrix_t t_yPowerBatch;
    Vector_t t_result;

    Vector_t t_inst;
    Vector_t t_y;
    Vector_t t_yPower;

};

typedef struct KernelPerceptron_st* KernelPerceptron_t;

struct PolynomialKernelPerceptron_st {
    float bias;
    int power;
};

typedef struct PolynomialKernelPerceptron_st* PolynomialKernelPerceptron_t;


struct SimplePerceptron_st {
    int c;
    int best_numit;

    Vector_t w;
    Vector_t w_avg;
    Vector_t best_w;
    Vector_t w_beta;
};

typedef struct SimplePerceptron_st* SimplePerceptron_t;


#endif