#include "PredictorGWCUDA.h"
#include "../config.h"

namespace {

	__device__ byte predict(byte *iData, unsigned w, unsigned h) {
		unsigned absolutePosition = threadIdx.x + blockIdx.x * THREADS;
		unsigned x = absolutePosition % w - 2;
		unsigned y = absolutePosition / w;
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

void PredictorGW::cudaPredictAll(void *diData, void *dPredicted, unsigned w, unsigned h){
	unsigned size = w * h;
	::predict<<<size/THREADS + 1, THREADS>>>(diData, dPredicted, w, h);
}
