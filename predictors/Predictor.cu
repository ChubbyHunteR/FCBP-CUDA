#include "Predictor.h"
#include "../config.h"

typedef unsigned char byte;
namespace cbpc_cuda{

	__device__ byte predict(byte *iData, unsigned* lookupOffsetX, unsigned* lookupOffsetY, unsigned w, unsigned h) {
//		unsigned absolutePosition = threadIdx.x + blockIdx.x * THREADS;
//		unsigned anchorX = absolutePosition % w;
//		unsigned anchorY = absolutePosition / w;
//		unsigned sum = 0;
//		unsigned x, y;
//		unsigned pixelHit = 0;
//
//		for(unsigned i = 0; i < N; ++i){
//			x = anchorX + lookupOffsetX[i];
//			y = anchorY  + lookupOffsetY[i];
//			if(x < w && y < h){
//				sum += iData[y * w + x];
//				++pixelHit;
//			}
//		}
//
//		return sum / pixelHit;

		unsigned absolutePosition = threadIdx.x + blockIdx.x * THREADS;
		unsigned x = absolutePosition % w;
		unsigned y = absolutePosition / w;
		if(x < w && y < h){
			return iData[y * w + x];
		}
		return 0;
	}

	__global__ void predict(void *diData, void *dPredicted, void* dLookupOffsetX, void* dLookupOffsetY, unsigned w, unsigned h) {
		byte* iData = (byte*) diData;
		byte* predicted = (byte*) dPredicted;
		unsigned* lookupOffsetX = (unsigned*) dLookupOffsetX;
		unsigned* lookupOffsetY = (unsigned*) dLookupOffsetY;

		predicted[threadIdx.x + blockIdx.x * THREADS] = predict(iData, lookupOffsetX, lookupOffsetY, w, h);
	}
}

void Predictor::predict(void *diData, void *dPredicted, void* dLookupOffsetX, void* dLookupOffsetY, unsigned w, unsigned h){
	unsigned size = w * h;
	cbpc_cuda::predict<<<size/THREADS + 1, THREADS>>>(diData, dPredicted, dLookupOffsetX, dLookupOffsetY, w, h);
}
