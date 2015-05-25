#include "PredictorGW.h"

byte PredictorGW::predict(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h) {
	x -= 2;
	int sum = 0;
	if(x < w && y < h){
		sum -= iData[y * w + x];
	}
	++x;
	if(x < w && y < h){
		sum += 2 * iData[y * w + x];
	}

	if(sum < 0){
		sum = 0;
	}else if(sum > 255){
		sum = 255;
	}
	return sum;
}
