#include "epblas/eputil.h"

const char* eparseGetErrorString(eparseError_t status) {

    switch (status) {

        case eparseSucess:
            return "Success";
        case eparseMemoryAllocationError:
            return "Memory allocation error occurred.";
        case eparseColumnNumberMissmatch:
            return "Number of columns in two matrix do not match.";
        case eparseNullPointer:
            return "Null pointer got";
        case eparseUnsupportedMemoryType:
            return "Unsupported Memory Type (Only libepblascuda supports CUDA memory allocation and only libepblasmkl supports MKL aligned memory allocation)";
        case eparseInvalidOperationRequest:
            return "Invalid operation request made. Remember that some calls might be invalid for CUDA. Such as setParallism() and  getMaxParallism()";
        case eparseKernelPerceptronDumpError:
            return "Error in dumping Kernel Perceptron model into file.";
        case eparseKernelPerceptronLoadError:
            return "Error in loading Kernel Perceptron model from file";
        case eparseKernelType:
            return "Unsupported Kernel Type";
        case eparseIndexOutofBound:
            return "Index out of bound";
		case eparseTooLargeCudaOp:
			return "Operation is beyond the limits of CUDA";
        default:
            return "Unknown error";
    }

}

