#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <ctime>

#include "PGMCBPD.h"
#include "../staticPredictors/PredictorN.h"
#include "../staticPredictors/PredictorNW.h"
#include "../staticPredictors/PredictorGW.h"
#include "../staticPredictors/PredictorW.h"
#include "../staticPredictors/PredictorNE.h"
#include "../staticPredictors/PredictorGN.h"
#include "../staticPredictors/PredictorPL.h"

string usage = " inputImageErrorFile...";

void fail(string msg){
	cerr<<msg<<endl;
	exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
	if(argc < 2){
		usage = string("Usage:\n") + argv[0] + usage;
		fail(usage);
	}

	for(unsigned fileOffset = 0; fileOffset < argc; fileOffset += FILESET_SIZE){
		vector<PGMImageError> inputImagesError;
		vector<PGMImage> outputImages;
		for(int i = 0; i < FILESET_SIZE && i < argc; ++i){
			if(i + fileOffset == 0){
				++i;
			}
			string inputName = argv[i + fileOffset];
			size_t dot = inputName.find_last_of('.');
			if(dot == inputName.npos){
				dot = inputName.length();
			}
			string outputName = inputName.substr(0, dot) + "_decoded" + inputName.substr(dot);

			inputImagesError.emplace_back(inputName.c_str());
			unsigned w = inputImagesError[i-1].getWidth();
			unsigned h = inputImagesError[i-1].getHeight();
			unsigned size = inputImagesError[i-1].getSize();
			unsigned maxPixel = inputImagesError[i-1].getPixelMax();
			outputImages.emplace_back(outputName.c_str(), w, h, maxPixel);
		}

		vector<Predictor*> predictors;
		predictors.push_back(new PredictorN);
		predictors.push_back(new PredictorNW);
		predictors.push_back(new PredictorGW);
		predictors.push_back(new PredictorW);
		predictors.push_back(new PredictorNE);
		predictors.push_back(new PredictorGN);
		predictors.push_back(new PredictorPL);

		PGMCBPD cbpd(inputImagesError, outputImages, predictors);
		cbpd.decode();
	}

	return 0;
}
