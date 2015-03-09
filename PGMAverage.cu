#include "PGMAverage.h"

PGMAverageCUDA::PGMAverageCUDA(PGMImage& input, PGMImage& output):input(input), output(output), w(input.getWidth()), h(input.getHeight(), size(input.getSize())){
	for(unsigned i = 0; i < N; ++i){
		lookupOffsetx[i] = -R + i % (2*R + 1);
		lookupOffsety[i] = -R + i / (2*R + 1);
	}

	iData = new byte[size];
	oData = new byte[size];
	input.readIntoArray(idata, 0, size);

	dData = NULL;
	CUDA_CHECK_RETURN(cudaMalloc( &dData, sizeof(byte) * size));
	CUDA_CHECK_RETURN(cudaMemcpy(dData, iData, sizeof(byte) * size, cudaMemcpyHostToDevice));
}

void PGMAverageCUDA::average(){
	average<<<size/THREADS + 1, THREADS>>>(dData);
//	for(unsigned y = 0; y < h; ++y){
//		for(unsigned x = 0; x < w; ++x){
//			output.writePixel(x, y, averagePixel(x, y));
//		}
//	}
	CUDA_CHECK_RETURN(cudaThreadSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy(oData, dData, sizeof(byte) * size, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree((void*) dData));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	for(unsigned j = 0; j < h; ++j){
		for(unsigned i = 0; i < w; ++i){
			output.writePixel(i, j, odata[i + j*w]);
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
		if(x >= 0 && x < w && y >= 0 && y < h){
			sum += input.getPixel(x, y);
			++pixelHit;
		}
	}

	return sum / pixelHit;
}

__global__ void PGMAverageCUDA::average(void *dData) {
	byte* data = (byte*) dData;
	data[threadIdx.x + blockIdx.x * THREADS] = invert(data[threadIdx.x + blockIdx.x * THREADS]);
}

__device__ byte PGMAverageCUDA::invert(byte b) {
	return -b;
}
