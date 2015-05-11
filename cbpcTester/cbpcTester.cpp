#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>

#include "../PGM/PGMImage.h"
#include "../PGM/PGMImageError.h"

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
	vector<PGMImage> predictionImages;
	vector<PGMImageError> errorImages;
	for(int i = 1; i < argc; ++i){
		string inputName = argv[i];
		size_t dot = inputName.find_last_of('.');
		if(dot == inputName.npos){
			dot = inputName.length();
		}
		string outputName = inputName.substr(0, dot) + "_prediction" + inputName.substr(dot);
		string errorName = inputName.substr(0, dot) + "_error" + inputName.substr(dot);

		inputImages.emplace_back(inputName.c_str());
		predictionImages.emplace_back(outputName.c_str());
		errorImages.emplace_back(errorName.c_str());
	}

	for(int i = 0; i < inputImages.size(); ++i){
		PGMImage& input = inputImages[i];
		PGMImage& prediction = predictionImages[i];
		PGMImageError& error = errorImages[i];
		for(unsigned p = 0; p < input.getSize(); ++p){
			if((int)prediction.getPixel(p) + (int)error.getPixel(p) != (int)input.getPixel(p)){
				cout << "Image " << i << " failed at pixel " << p << endl;
				break;
			}
		}
		cout << "Image " << i << " done" << endl;
	}

	return 0;
}
