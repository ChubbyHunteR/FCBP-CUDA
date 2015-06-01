#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <string>
#include <ctime>

#include "GRCoder.h"

using namespace std;

string usage = " inputImageError.pgm...";

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
		vector<PGMImageError> inputs;
		vector<ofstream*> outputs;
		vector<GRCoder> coders;
		for(int i = 1; i < FILESET_SIZE && i < argc; ++i){
			string inputName = argv[i + fileOffset];
			size_t dot = inputName.find_last_of('.');
			if(dot == inputName.npos){
				dot = inputName.length();
			}
			string outputName = inputName.substr(0, dot) + "_encoded" + inputName.substr(dot);

			inputs.emplace_back(inputName.c_str());
			unsigned w = inputs[i-1].getWidth();
			unsigned h = inputs[i-1].getHeight();
			unsigned size = inputs[i-1].getSize();
			unsigned maxPixel = inputs[i-1].getPixelMax();
			outputs.emplace_back(new ofstream(outputName, ios_base::out | ios_base::binary));
			coders.emplace_back(inputs[i-1], *outputs[i-1]);
		}

		for(auto& coder : coders){
			coder.encode();
		}

		for(auto output : outputs){
			delete output;
		}
	}

	return 0;
}
