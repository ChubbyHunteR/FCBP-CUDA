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

	vector<ifstream*> inputs;
	vector<GRDecoder> decoders;
	for(int i = 1; i < argc; ++i){
		string inputName = argv[i];
		size_t dot = inputName.find_last_of('.');
		if(dot == inputName.npos){
			dot = inputName.length();
		}
		string outputName = inputName.substr(0, dot) + "_decoded" + inputName.substr(dot);

		inputs.emplace_back(new ifstream(inputName, ios_base::in | ios_base::binary));
		decoders.emplace_back(*inputs.back(), outputName);
	}

	for(auto& decoder : decoders){
		decoder.decode();
	}

	for(auto input : inputs){
		delete input;
	}

	return 0;
}
