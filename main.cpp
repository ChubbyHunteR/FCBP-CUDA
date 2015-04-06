#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <ctime>
#include "PGMImage.h"
#include "PGMAverage.h"
#include "PGMCBPCCUDA.h"
#include "predictors/PredictorN.h"
#include "predictors/PredictorNW.h"
#include "predictors/PredictorGW.h"
#include "predictors/PredictorW.h"
#include "predictors/PredictorNE.h"
#include "predictors/PredictorGN.h"
#include "predictors/PredictorPL.h"
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

	PredictorN predictorN;
	PredictorNW predictorNW;
	PredictorGW predictorGW;
	PredictorW predictorW;
	PredictorNE predictorNE;
	PredictorGN predictorGN;
	PredictorPL predictorPL;

	PGMCBPCCUDA cbpc(picInput, picOutput);
	cbpc.addPredictor(&predictorN);
	cbpc.addPredictor(&predictorNW);
	cbpc.addPredictor(&predictorGW);
	cbpc.addPredictor(&predictorW);
	cbpc.addPredictor(&predictorNE);
	cbpc.addPredictor(&predictorGN);
	cbpc.addPredictor(&predictorPL);
	cbpc.init();
	cbpc.predict();

	return 0;
}
