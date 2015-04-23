#include "predictors.h"

__device__ unsigned numOfPredictors = 7;

__device__ byte (* predictors[])(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h) =
{
	predictGN,
	predictGW,
	predictN,
	predictNE,
	predictNW,
	predictPL,
	predictW,
	NULL
};

__device__ byte predictGN(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h) {
	y -= 2;
	int sum = 0;
	if(x < w && y < h){
		sum -= iData[y * w + x];
	}
	++y;
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

__device__ byte predictGW(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h) {
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

__device__ byte predictN(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h) {
	--y;
	if(x < w && y < h){
		return iData[y * w + x];
	}
	return 0;
}

__device__ byte predictNE(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h) {
	++x;
	--y;
	if(x < w && y < h){
		return iData[y * w + x];
	}
	return 0;
}

__device__ byte predictNW(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h) {
	--x;
	--y;
	if(x < w && y < h){
		return iData[y * w + x];
	}
	return 0;
}

__device__ byte predictPL(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h) {
	--x;
	--y;
	short sum = 0;
	if(x < w && y < h){
		sum -= iData[y * w + x];
	}
	++x;
	if(x < w && y < h){
		sum += iData[y * w + x];
	}
	--x;
	++y;
	if(x < w && y < h){
		sum += iData[y * w + x];
	}

	if(sum < 0){
		sum = 0;
	}else if(sum > 255){
		sum = 255;
	}
	return sum;
}

__device__ byte predictW(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h) {
	--x;
	if(x < w && y < h){
		return iData[y * w + x];
	}
	return 0;
}
