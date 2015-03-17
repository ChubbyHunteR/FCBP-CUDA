#include <iostream>
#include <cstdlib>
#include <string>
#include "PGMImage.h"
#include "PGMAverage.h"
#include "PGMAverageCUDA.h"
#include <ctime>

#define LOOP 10

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

	PGMAverage average1(picInput, picOutput1);
	clock_t t1 = clock();
	for(int i = 0; i < LOOP; ++i){
		average1.average();
	}
	t1 = clock() - t1;
	cout<<"CPU: "<<(float) t1 / CLOCKS_PER_SEC * 1000<<" ms"<<endl;

	PGMAverageCUDA average2(picInput, picOutput2);
	clock_t t2 = clock();
	for(int i = 0; i < LOOP; ++i){
		average2.average();
	}
	t2 = clock() - t2;
	cout<<"GPU: "<<(float) t2 / CLOCKS_PER_SEC * 1000<<" ms"<<endl;

	return 0;
}
