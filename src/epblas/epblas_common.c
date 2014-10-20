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

