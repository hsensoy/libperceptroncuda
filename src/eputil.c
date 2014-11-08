#include "epblas/eputil.h"

char* eparseGetErrorString(eparseError_t status) {

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

eparseError_t newProgress(Progress_t *p, const char *unit, int total, double report_rate) {

    *p = malloc(sizeof (struct Progress_st));

    if (*p == NULL)
        return eparseMemoryAllocationError;

    (*p)->current = 0;
    (*p)->req_tick_per_report = MAX(MIN(MAX(report_rate, 0.001), 1.) * total, MIN_REQ_TICK_PER_REPORT);

    (*p)->unit = strdup(unit);

    log_info ("Report for every %d %s\n", (*p)->req_tick_per_report, (*p)->unit);
    (*p)->total = total;

    gettimeofday(&((*p)->last_report_epoch), NULL);
    (*p)->last_tick_count = 0;

    return eparseSucess;

}

eparseError_t deleteProgress(Progress_t p){
    free(p->unit);
    free(p);


    return eparseSucess;
}

bool tickProgress(Progress_t p) {

    if (p->current < p->total) {
        (p->current)++;
        struct timeval current_epoch, timerElapsed;

        gettimeofday(&current_epoch, NULL);
        timersub(&current_epoch, &( p->last_report_epoch), &timerElapsed);

        double elapsed_sec = timerElapsed.tv_sec +timerElapsed.tv_usec/1000000.0;
        if (p->current == p->total) {

            log_info("100.00%% of %d %s are done", p->current, p->unit);

            log_info( "%d more %s in %lf sec.", p->current - p->last_tick_count, p->unit, elapsed_sec);

            gettimeofday(&(p->last_report_epoch), NULL);

            p->last_tick_count = p->current;

            return true;
        }
        else if (p->current % p->req_tick_per_report == 0) {

            log_info( "%.2f%%(%d) of %s are done", ((p->current * 1.) * 100.) / (p->total), p->current,p->unit);

            log_info( "%d more %s in %lf sec.", p->current - p->last_tick_count, p->unit, elapsed_sec);

            gettimeofday(&(p->last_report_epoch), NULL);

            p->last_tick_count = p->current;

            return true;
        }
    }

    return false;

}
