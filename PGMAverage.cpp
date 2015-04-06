#include "PGMAverage.h"

PGMAverage::PGMAverage(PGMImage& input, PGMImage& output):input(input), output(output), w(input.getWidth()), h(input.getHeight()), size(input.getSize()){
	for(int i = 0; i < R_A; ++i){
		lookupOffsetx[i] = -R + i % (2*R + 1);
		lookupOffsety[i] = -R + i / (2*R + 1);
	}
}

void PGMAverage::average(){
	for(unsigned y = 0; y < h; ++y){
		for(unsigned x = 0; x < w; ++x){
			output.writePixel(x, y, averagePixel(x, y));
		}
	}
}

byte PGMAverage::averagePixel(unsigned anchorx, unsigned anchory){
	unsigned sum = 0;
	unsigned x, y;
	unsigned pixelHit = 0;

	for(unsigned i = 0; i < R_A; ++i){
		x = anchorx + lookupOffsetx[i];
		y = anchory + lookupOffsety[i];
		if(x < w && y < h){
			sum += input.getPixel(x, y);
			++pixelHit;
		}
	}

	return sum / pixelHit;
}
