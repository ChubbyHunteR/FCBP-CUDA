#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <ctime>
#include "PGMImage.h"
#include "PGMCBPDCUDA.h"
#include "predictors/PredictorN.h"
#include "predictors/PredictorNW.h"
#include "predictors/PredictorGW.h"
#include "predictors/PredictorW.h"
#include "predictors/PredictorNE.h"
#include "predictors/PredictorGN.h"
#include "predictors/PredictorPL.h"
#include "config.h"

string usage = " inputImage.pgm...";

void fail(string msg){
	cerr<<msg<<endl;
	exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
	if(argc < 2){
		usage = string("Usage:\n") + argv[0] + usage;
		fail(usage);
	}

	vector<PGMImage> inputImages;
	vector<PGMImage> outputImages;
	vector<PGMImage> errorImages;
	for(int i = 1; i < argc; ++i){
		string inputName = argv[i];
		size_t dot = inputName.find_last_of('.');
		if(dot == inputName.npos){
			dot = inputName.length();
		}
		string outputName = inputName.substr(0, dot) + "_prediction" + inputName.substr(dot);
		string errorName = inputName.substr(0, dot) + "_error" + inputName.substr(dot);

		inputImages.emplace_back(inputName.c_str());
		unsigned w = inputImages[i-1].getWidth();
		unsigned h = inputImages[i-1].getHeight();
		unsigned size = inputImages[i-1].getSize();
		unsigned maxPixel = inputImages[i-1].getPixelMax();
		outputImages.emplace_back(outputName.c_str(), w, h, maxPixel);
		errorImages.emplace_back(errorName.c_str(), w, h, maxPixel);
	}

	vector<Predictor*> predictors;
	predictors.push_back(new PredictorN);
	predictors.push_back(new PredictorNW);
	predictors.push_back(new PredictorGW);
	predictors.push_back(new PredictorW);
	predictors.push_back(new PredictorNE);
	predictors.push_back(new PredictorGN);
	predictors.push_back(new PredictorPL);

	PGMCBPDCUDA cbpd(inputImages, outputImages, errorImages, predictors);
	cbpd.predict();

	return 0;
}
