#include <stdio.h>
#include <stdlib.h>
#include "perceptron.h"
#include "debug.h"

void creatndDrop() {
    Perceptron_t pkp = newPolynomialKernelPerceptron(4, 1.);

    EPARSE_CHECK_RETURN(deletePerceptron(pkp))
}

int main() {
    creatndDrop();
}
