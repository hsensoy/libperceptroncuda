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
};

typedef struct KernelPerceptron_st* KernelPerceptron_t;

struct PolynomialKernelPerceptron_st {
    float bias;
    int power;
};

typedef struct PolynomialKernelPerceptron_st* PolynomialKernelPerceptron_t;




#endif