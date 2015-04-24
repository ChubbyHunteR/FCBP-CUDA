#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <ctime>
#include "PGMImage.h"
#include "PGMImageError.h"
#include "PGMCBPDCUDA.h"
#include "config.h"

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

	vector<PGMImageError> inputImagesError;
	vector<PGMImage> outputImages;
	vector<PGMImage> predictionImages;
	for(int i = 1; i < argc; ++i){
		string inputName = argv[i];
		size_t dot = inputName.find_last_of('.');
		if(dot == inputName.npos){
			dot = inputName.length();
		}
		string outputName = inputName.substr(0, dot) + "_decoded" + inputName.substr(dot);
		string predictionName = inputName.substr(0, dot) + "_prediction" + inputName.substr(dot);

		inputImagesError.emplace_back(inputName.c_str());
		unsigned w = inputImagesError[i-1].getWidth();
		unsigned h = inputImagesError[i-1].getHeight();
		unsigned size = inputImagesError[i-1].getSize();
		unsigned maxPixel = inputImagesError[i-1].getPixelMax();
		outputImages.emplace_back(outputName.c_str(), w, h, maxPixel);
		predictionImages.emplace_back(predictionName.c_str(), w, h, maxPixel);
	}

	PGMCBPDCUDA cbpd(inputImagesError, outputImages, predictionImages);
	cbpd.decode();

	return 0;
}
