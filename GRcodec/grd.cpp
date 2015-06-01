#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <string>
#include <ctime>

#include "GRDecoder.h"

using namespace std;

string usage = " inputImageErrorEncoded.pgm...";

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
		vector<ifstream*> inputs;
		vector<GRDecoder> decoders;
		for(int i = 1; i < FILESET_SIZE && i < argc; ++i){
			if(i + fileOffset == 0){
				++i;
			}
			string inputName = argv[i + fileOffset];
			size_t dot = inputName.find_last_of('.');
			if(dot == inputName.npos){
				dot = inputName.length();
			}
			string outputName = inputName.substr(0, dot) + "_decoded" + inputName.substr(dot);

			inputs.emplace_back(new ifstream(inputName, ios_base::in | ios_base::binary));
			decoders.emplace_back(*inputs[i-1], outputName);
		}

		for(auto& decoder : decoders){
			decoder.decode();
		}

		for(auto input : inputs){
			delete input;
		}
	}

	return 0;
}
