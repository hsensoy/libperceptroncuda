/*
 * eparseutil.h
 *
 *  Created on: Sep 10, 2014
 *      Author: husnusensoy
 */

#ifndef EPARSEUTIL_H_
#define EPARSEUTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "debug.h"
#include <sys/time.h>
#include <time.h>

#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#define MIN_REQ_TICK_PER_REPORT 10

typedef int edim_t ;
typedef long xedim_t;


enum eparseError_t {
    eparseSucess,
    eparseFailOthers,
    eparseMemoryAllocationError,
    eparseColumnNumberMissmatch,
    eparseNullPointer,
    eparseUnsupportedMemoryType,
    eparseIndexOutofBound,
    eparseInvalidOperationRequest,
    eparseKernelType,
    eparseKernelPerceptronLoadError,
    eparseKernelPerceptronDumpError
};

typedef enum eparseError_t eparseError_t;

char* eparseGetErrorString(eparseError_t status);

#define EPARSE_CHECK_RETURN(v) {											\
		eparseError_t stat = v;										\
		if (stat != eparseSucess) {										\
			fprintf(stderr, "Error %s at line %d in file %s\n",					\
					eparseGetErrorString(stat), __LINE__, __FILE__);		\
			exit(1);															\
		} }


struct Progress_t {
    char *unit;
    int total;
    int current;
    int req_tick_per_report;

    struct timeval last_report_epoch;
    int last_tick_count;
};

typedef struct Progress_t* Progress_t;

/**
*
* @param p Reference to an Progress_t
* @param unit Name of the unit
* @param total Total number of ticks to be reached
* @param report_rate Reporting rate. If reporting rate if 0.05, for every 5% progress over total ticks will be reported.
* @return eparseSucess if creating Progress_t is succeed.
*/
eparseError_t newProgress(Progress_t *p, const char *unit, int total, double report_rate);

eparseError_t deleteProgress(Progress_t p);

/**
*
* @param p Progress tracking object
* @return Reporting indicator. true if tickProgress reached reporting threshold. Otherwise false.
*/
bool tickProgress(Progress_t p);


#endif /* EPARSEUTIL_H_ */
