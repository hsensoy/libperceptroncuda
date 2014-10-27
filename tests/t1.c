#include <stdio.h>
#include <stdlib.h>
#include "perceptron.h"
#include "debug.h"

void creatndDrop() {
	log_info("Creating polynomial kernel perceptron");
    Perceptron_t pkp = newPolynomialKernelPerceptron(4, 1.);

	log_info("Deleting polynomial kernel perceptron");
    EPARSE_CHECK_RETURN(deletePerceptron(pkp))
}

int main() {
    creatndDrop();
}
