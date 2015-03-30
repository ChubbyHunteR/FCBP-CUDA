#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <ctime>
#include "PGMImage.h"
#include "PGMAverage.h"
#include "PGMCBPCCUDA.h"
#include "predictors/Predictor.h"
#include "config.h"

string usage = " inputImage.pgm outputImage.pgm";

void fail(string msg){
	cerr<<msg<<endl;
	exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
	if(argc != 3){
		usage = string("Usage:\n") + argv[0] + usage;
		fail(usage);
	}

	PGMImage picInput(argv[1]);
	unsigned w = picInput.getWidth();
	unsigned h = picInput.getHeight();
	unsigned size = picInput.getSize();
	PGMImage picOutput(argv[2], w, h, picInput.getPixelMax());

	Predictor predictor;

	PGMCBPCCUDA cbpc(picInput, picOutput);
	cbpc.addPredictor(&predictor);
	cbpc.init();
	cbpc.getStaticPrediction(0);

	return 0;
}
