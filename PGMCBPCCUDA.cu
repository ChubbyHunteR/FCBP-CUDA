#include "PGMCBPCCUDA.h"

#define SWAP(a, b) (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))

namespace {
	__device__ void insert(unsigned dist, unsigned x, unsigned y, unsigned similarPixelsX[], unsigned similarPixelsY[], unsigned similarPixelsDistance[], unsigned* numOfSimilarPixels){
		unsigned tmpDist = dist;
		unsigned tmpPixelX = x;
		unsigned tmpPixelY = y;

		for(int i = 0; i < *numOfSimilarPixels; ++i){
			if(tmpDist > similarPixelsDistance[i]){
				SWAP(tmpDist, similarPixelsDistance[i]);
				SWAP(tmpPixelX, similarPixelsX[i]);
				SWAP(tmpPixelY, similarPixelsY[i]);
			}
		}

		if(*numOfSimilarPixels < M){
			similarPixelsDistance[*numOfSimilarPixels] = tmpDist;
			similarPixelsX[*numOfSimilarPixels] = tmpPixelX;
			similarPixelsY[*numOfSimilarPixels] = tmpPixelY;
			++*numOfSimilarPixels;
		}
	}

	__device__ unsigned distance(byte* iData, unsigned anchorX, unsigned anchorY, unsigned x, unsigned y, unsigned* vectorOffsetx, unsigned* vectorOffsety, unsigned w, unsigned h){
		unsigned x1, x2, y1, y2, sum = 0;
		int b1, b2;
		for(int j = 0; j < D; ++j){
			for(int i = 0; i < D; ++i){
				x1 = anchorX + vectorOffsetx[i];
				y1 = anchorY + vectorOffsety[j];
				x2 = x + vectorOffsetx[i];
				y2 = y + vectorOffsety[j];
				b1 = b2 = 0;
				if(x1 < w && y1 < h){
					b1 = iData[x1 + y1 * w];
				}
				if(x2 < w && y2 < h){
					b2 = iData[x2 + y2 * w];
				}
				sum += (b1-b2) * (b1-b2);
			}
		}
		return sum;
	}

	__device__ byte predict(byte* iData, byte** predicted, unsigned numOfPredictors, unsigned* radiusOffsetx, unsigned* radiusOffsety, unsigned* vectorOffsetx, unsigned* vectorOffsety, unsigned w, unsigned h) {
		unsigned absolutePosition = threadIdx.x + blockIdx.x * THREADS;
		unsigned anchorX = absolutePosition % w;
		unsigned anchorY = absolutePosition / w;
		unsigned x, y, dist, numOfSimilarPixels = 0;
		unsigned similarPixelsDistance[M];
		unsigned similarPixelsX[M];
		unsigned similarPixelsY[M];
		float penalties[MAX_PREDICTORS];

		for(int j = 0; j < R_A; ++j){
			for(int i = 0; i < R_A; ++i){
				x = anchorX + radiusOffsetx[i];
				y = anchorY + radiusOffsety[j];
				if(x < w && y < h){
					dist = distance(iData, anchorX, anchorY, x, y, vectorOffsetx, vectorOffsety, w, h);
					insert(dist, x, y, similarPixelsX, similarPixelsY, similarPixelsDistance, &numOfSimilarPixels);
				}
			}
		}

		for(int i = 0; i < numOfPredictors; ++i){
			unsigned sum = 0;
			for(int j = 0; j < numOfSimilarPixels; ++j){
				int prediction = predicted[i][ similarPixelsX[j] + similarPixelsY[j] * w ];
				int pixel = iData[ similarPixelsX[j] + similarPixelsY[j] * w ];
				int diff = prediction - pixel;
				sum += diff*diff;
			}
			penalties[i] = (float)sum / numOfSimilarPixels;
		}

		float sum = 0;
		float penaltiesSum = 0;
		for(int i = 0; i < numOfPredictors; ++i){
			sum += predicted[i][absolutePosition] / penalties[i];
			penaltiesSum += 1 / penalties[i];
		}

		return sum / penaltiesSum;
	}

	__global__ void predict(void* diData, void* doData, void** dPredicted, unsigned numOfPredictors, void* dRadiusOffsetx, void* dRadiusOffsety, void* dVectorOffsetx, void* dVectorOffsety, unsigned w, unsigned h) {
		unsigned absolutePosition = threadIdx.x + blockIdx.x * THREADS;
		if(absolutePosition >= w*h){
			return;
		}

		byte* iData = (byte*) diData;
		byte* oData = (byte*) doData;
		byte** predicted = (byte**) dPredicted;
		unsigned* radiusOffsetx = (unsigned*) dRadiusOffsetx;
		unsigned* radiusOffsety = (unsigned*) dRadiusOffsety;
		unsigned* vectorOffsetx = (unsigned*) dVectorOffsetx;
		unsigned* vectorOffsety = (unsigned*) dVectorOffsety;

		oData[absolutePosition] = predict(iData, predicted, numOfPredictors, radiusOffsetx, radiusOffsety, vectorOffsetx, vectorOffsety, w, h);
	}
}

PGMCBPCCUDA::PGMCBPCCUDA(PGMImage& input, PGMImage& output):input(input), output(output), numOfPredictors(0), w(input.getWidth()), h(input.getHeight()), size(input.getSize()){
	for(int i = 0; i < R_A; ++i){
		radiusOffsetx[i] = i % (2*R + 1) - R;
		radiusOffsety[i] = i / (2*R + 1) - R;
	}
	for(int i = 0; i < D; ++i){
		vectorOffsetx[i] = i % (D_R + 2) - D_R;
		vectorOffsety[i] = i / (D_R + 2) - D_R;
	}
}

PGMCBPCCUDA::~PGMCBPCCUDA(){
	CUDA_CHECK_RETURN(cudaFree((void*) dRadiusOffsetx));
	CUDA_CHECK_RETURN(cudaFree((void*) dRadiusOffsety));
	CUDA_CHECK_RETURN(cudaFree((void*) dVectorOffsetx));
	CUDA_CHECK_RETURN(cudaFree((void*) dVectorOffsety));
	CUDA_CHECK_RETURN(cudaFree((void*) diData));
	CUDA_CHECK_RETURN(cudaFree((void*) doData));
	CUDA_CHECK_RETURN(cudaDeviceReset());
}

void PGMCBPCCUDA::init(){
	iData = new byte[size];
	oData = new byte[size];
	input.readIntoArray(iData, 0, size);

	doData = NULL;
	diData = NULL;
	CUDA_CHECK_RETURN(cudaMalloc(&diData, sizeof(byte) * size));
	CUDA_CHECK_RETURN(cudaMalloc(&doData, sizeof(byte) * size));
	CUDA_CHECK_RETURN(cudaMemcpy(diData, iData, sizeof(byte) * size, cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc(&dRadiusOffsetx, sizeof(radiusOffsetx)));
	CUDA_CHECK_RETURN(cudaMalloc(&dRadiusOffsety, sizeof(radiusOffsety)));
	CUDA_CHECK_RETURN(cudaMemcpy(dRadiusOffsetx, radiusOffsetx, sizeof(radiusOffsetx), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(dRadiusOffsety, radiusOffsety, sizeof(radiusOffsety), cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc(&dVectorOffsetx, sizeof(vectorOffsetx)));
	CUDA_CHECK_RETURN(cudaMalloc(&dVectorOffsety, sizeof(vectorOffsety)));
	CUDA_CHECK_RETURN(cudaMemcpy(dVectorOffsetx, vectorOffsetx, sizeof(vectorOffsetx), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(dVectorOffsety, vectorOffsety, sizeof(vectorOffsety), cudaMemcpyHostToDevice));

	for(int i = 0; i < numOfPredictors; ++i){
		CUDA_CHECK_RETURN(cudaMalloc(&dPredicted[i], sizeof(short) * size));
	}
	for(int i = 0; i < numOfPredictors; ++i){
		predictor[i]->predict(diData, dPredicted[i], w, h);
	}
}

void PGMCBPCCUDA::addPredictor(Predictor* predictor){
	if(numOfPredictors + 1 >= MAX_PREDICTORS){
		return;
	}

	this->predictor[numOfPredictors++] = predictor;
}

void PGMCBPCCUDA::predict(){
	::predict<<<size/THREADS + 1, THREADS>>>(diData, diData, dPredicted, numOfPredictors, dRadiusOffsetx, dRadiusOffsety, dVectorOffsetx, dVectorOffsety, w, h);

	CUDA_CHECK_RETURN(cudaMemcpy(oData, doData, sizeof(byte) * size, cudaMemcpyDeviceToHost));

	for(unsigned j = 0; j < h; ++j){
		for(unsigned i = 0; i < w; ++i){
			output.writePixel(i, j, oData[i + j*w]);
		}
	}
}

void PGMCBPCCUDA::getStaticPrediction(unsigned i){
	if(i >= numOfPredictors){
		return;
	}

	CUDA_CHECK_RETURN(cudaMemcpy(oData, dPredicted[i], sizeof(byte) * size, cudaMemcpyDeviceToHost));

	for(unsigned j = 0; j < h; ++j){
		for(unsigned i = 0; i < w; ++i){
			output.writePixel(i, j, oData[i + j*w]);
		}
	}
}
