#include "PredictorW.h"
#include "../config.h"

namespace {

	__device__ byte predict(byte *iData, unsigned w, unsigned h) {
		unsigned absolutePosition = threadIdx.x + blockIdx.x * THREADS;
		unsigned x = absolutePosition % w - 1;
		unsigned y = absolutePosition / w;
		if(x < w && y < h){
			return iData[y * w + x];
		}
		return 0;
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

void PredictorW::cudaPredictAll(void *diData, void *dPredicted, unsigned w, unsigned h){
	unsigned size = w * h;
	::predict<<<size/THREADS + 1, THREADS>>>(diData, dPredicted, w, h);
}
