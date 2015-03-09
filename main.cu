#include <iostream>
#include <cstdlib>
#include <string>
#include "PGMImage.h"
#include "PGMAverage.h"

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

	PGMAverageCUDA average(picInput, picOutput);
	average.average();

	return 0;
}
