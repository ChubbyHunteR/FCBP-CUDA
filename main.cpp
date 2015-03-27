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
	string outName(argv[2]);
	PGMImage picOutput1((outName + "1").c_str(), w, h, picInput.getPixelMax());
	PGMImage picOutput2((outName + "2").c_str(), w, h, picInput.getPixelMax());

	cout<<"Radius: "<<R<<endl;
	cout<<"Threads: "<<THREADS<<endl;
	cout<<"Loops: "<<LOOP<<endl;

	PGMAverage average1(picInput, picOutput1);
	clock_t t1 = clock();
	for(int i = 0; i < LOOP; ++i){
//		average1.average();
	}
	t1 = clock() - t1;
	cout<<"CPU Time: "<<(float) t1 / CLOCKS_PER_SEC<<" s"<<endl;

	PGMCBPCCUDA average2(picInput, picOutput2);
	clock_t t2 = clock();
	for(int i = 0; i < LOOP; ++i){
		average2.average();
	}
	t2 = clock() - t2;
	cout<<"GPU time: "<<(float) t2 / CLOCKS_PER_SEC<<" s"<<endl;

	return 0;
}
