#include "epblas/epblas.h"

void printMatrix(const char* heading, Matrix_t m, FILE *fp) {

    long r, c;
    if (fp != NULL) {
        if (heading != NULL)
            fprintf(fp, "%s\n", heading);
        for (r = 0; r < m->nrow; r++) {
            for (c = 0; c < m->ncol; c++) {
                fprintf(fp, "%f", m->data[r * m->ncol + c]);
                if (c < m->ncol - 1)
                    fprintf(fp, "\t");
            }
            fprintf(fp, "\n");
        }

        fprintf(fp, "\n");
    }
}

static char *temp_buffer = NULL;

// TODO: Single call at a time for the time being.
char* humanreadable_size(size_t bytes) {
    if (temp_buffer == NULL) {
        temp_buffer = (char*) malloc(1024);
        check(temp_buffer != NULL, "Memory allocation error");
    }

    if (bytes >= 1024 * 1024 * 1024) {
  
        sprintf(temp_buffer, "%.3f GB", bytes / 1024. / 1024. / 1024.);
    } else if (bytes >= 1024 * 1024) {
        sprintf(temp_buffer, "%.3f MB", bytes / 1024. / 1024.);
    } else if (bytes >= 1024) {
        sprintf(temp_buffer, "%.3f KB", (bytes / 1024.));
    } else{
        sprintf(temp_buffer, "%.3f B", bytes/1.);
        }

    return temp_buffer;

    error: exit(1);
}

