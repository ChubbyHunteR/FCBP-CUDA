#include "PGMAverageCUDA.h"

__device__ byte average(byte *iData, unsigned* lookupOffsetX, unsigned* lookupOffsetY, unsigned w, unsigned h) {
	unsigned absolutePosition = threadIdx.x + blockIdx.x * THREADS;
	unsigned anchorX = absolutePosition % w;
	unsigned anchorY = absolutePosition / w;
	unsigned sum = 0;
	unsigned x, y;
	unsigned pixelHit = 0;

	for(unsigned i = 0; i < N; ++i){
		x = anchorX + lookupOffsetX[i];
		y = anchorY  + lookupOffsetY[i];
		if(x < w && y < h){
			sum += iData[y * w + x];
			++pixelHit;
		}
	}

	return sum / pixelHit;
}

__global__ void average(void *diData, void *doData, void* dLookupOffsetX, void* dLookupOffsetY, unsigned w, unsigned h) {
	byte* iData = (byte*) diData;
	byte* oData = (byte*) doData;
	unsigned* lookupOffsetX = (unsigned*) dLookupOffsetX;
	unsigned* lookupOffsetY = (unsigned*) dLookupOffsetY;

	oData[threadIdx.x + blockIdx.x * THREADS] = average(iData, lookupOffsetX, lookupOffsetY, w, h);
}

PGMAverageCUDA::PGMAverageCUDA(PGMImage& input, PGMImage& output):input(input), output(output), w(input.getWidth()), h(input.getHeight()), size(input.getSize()){
	for(int i = 0; i < N; ++i){
		lookupOffsetx[i] = -R + i % (2*R + 1);
		lookupOffsety[i] = -R + i / (2*R + 1);
	}

	iData = new byte[size];
	oData = new byte[size];
	input.readIntoArray(iData, 0, size);

	doData = NULL;
	diData = NULL;
	CUDA_CHECK_RETURN(cudaMalloc(&diData, sizeof(byte) * size));
	CUDA_CHECK_RETURN(cudaMalloc(&doData, sizeof(byte) * size));
	CUDA_CHECK_RETURN(cudaMemcpy(diData, iData, sizeof(byte) * size, cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc(&dLookupOffsetx, sizeof(lookupOffsetx)));
	CUDA_CHECK_RETURN(cudaMalloc(&dLookupOffsety, sizeof(lookupOffsety)));
	CUDA_CHECK_RETURN(cudaMemcpy(dLookupOffsetx, lookupOffsetx, sizeof(lookupOffsetx), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(dLookupOffsety, lookupOffsety, sizeof(lookupOffsety), cudaMemcpyHostToDevice));
}

PGMAverageCUDA::~PGMAverageCUDA(){
	CUDA_CHECK_RETURN(cudaFree((void*) diData));
	CUDA_CHECK_RETURN(cudaFree((void*) doData));
	CUDA_CHECK_RETURN(cudaFree((void*) dLookupOffsetx));
	CUDA_CHECK_RETURN(cudaFree((void*) dLookupOffsety));
	CUDA_CHECK_RETURN(cudaDeviceReset());
}

void PGMAverageCUDA::average(){
	::average<<<size/THREADS + 1, THREADS>>>(diData, doData, dLookupOffsetx, dLookupOffsety, w, h);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy(oData, doData, sizeof(byte) * size, cudaMemcpyDeviceToHost));

	for(unsigned j = 0; j < h; ++j){
		for(unsigned i = 0; i < w; ++i){
			output.writePixel(i, j, oData[i + j*w]);
		}
	}
}

byte PGMAverageCUDA::averagePixel(unsigned anchorx, unsigned anchory){
	unsigned sum = 0;
	unsigned x, y;
	unsigned pixelHit = 0;

	for(unsigned i = 0; i < N; ++i){
		x = anchorx + lookupOffsetx[i];
		y = anchory + lookupOffsety[i];
		if(x < w && y < h){
			sum += input.getPixel(x, y);
			++pixelHit;
		}
	}

	return sum / pixelHit;
}
