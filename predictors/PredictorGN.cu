#include "PredictorGN.h"
#include "../config.h"

typedef unsigned char byte;
namespace {

	__device__ byte predict(byte *iData, unsigned w, unsigned h) {
		unsigned absolutePosition = threadIdx.x + blockIdx.x * THREADS;
		unsigned x = absolutePosition % w - 2;
		unsigned y = absolutePosition / w;
		short sum = 0;
		if(x < w && y < h){
			sum -= iData[y * w + x];
		}
		++x;
		if(x < w && y < h){
			sum += 2 * iData[y * w + x];
		}
		return sum;
	}

	__global__ void predict(void *diData, void *dPredicted, unsigned w, unsigned h) {
		unsigned absolutePosition = threadIdx.x + blockIdx.x * THREADS;
		if(absolutePosition >= w*h){
			return;
		}

		byte* iData = (byte*) diData;
		byte* predicted = (byte*) dPredicted;

		predicted[absolutePosition] = predict(iData, w, h);
	}
}

void PredictorGN::predict(void *diData, void *dPredicted, unsigned w, unsigned h){
	unsigned size = w * h;
	::predict<<<size/THREADS + 1, THREADS>>>(diData, dPredicted, w, h);
}
