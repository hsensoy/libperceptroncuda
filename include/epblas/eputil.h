/*
 * eparseutil.h
 *
 *  Created on: Sep 10, 2014
 *      Author: husnusensoy
 */

#ifndef EPUTIL_H_
#define EPUTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "debug.h"
#include "util.h"
#include <sys/time.h>
#include <time.h>



typedef int edim_t ;
typedef long xedim_t;


enum eparseError_t {
    eparseSucess=0,
    eparseFailOthers,
    eparseMemoryAllocationError,
    eparseColumnNumberMissmatch,
    eparseNullPointer,
    eparseUnsupportedMemoryType,
    eparseInvalidOperationRequest,
	eparseIndexOutofBound,
    eparseKernelType,
    eparseKernelPerceptronLoadError,
    eparseKernelPerceptronDumpError,
	eparseTooLargeCudaOp,
	eparseNotImplementedYet
};

typedef enum eparseError_t eparseError_t;

const char* eparseGetErrorString(eparseError_t status);

#define EPARSE_CHECK_RETURN(v) {											\
		eparseError_t stat = v;										\
		if (stat != eparseSucess) {										\
			fprintf(stderr, "Error %s at line %d in file %s\n",					\
					eparseGetErrorString(stat), __LINE__, __FILE__);		\
			exit(1);															\
		} }
		
#endif /* EPARSEUTIL_H_ */
