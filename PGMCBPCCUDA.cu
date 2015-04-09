#include "PGMCBPCCUDA.h"
#include <iostream>

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
		int pix1, pix2;
		for(int i = 0; i < D; ++i){
			x1 = anchorX + vectorOffsetx[i];
			y1 = anchorY + vectorOffsety[i];
			x2 = x + vectorOffsetx[i];
			y2 = y + vectorOffsety[i];
			pix1 = pix2 = 0;
			if(x1 < w && y1 < h){
				pix1 = iData[x1 + y1 * w];
			}
			if(x2 < w && y2 < h){
				pix2 = iData[x2 + y2 * w];
			}
			sum += (pix1-pix2) * (pix1-pix2);
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
		float* penalties = new float[numOfPredictors];

		for(int i = 0; i < R_A; ++i){
			x = anchorX + radiusOffsetx[i];
			y = anchorY + radiusOffsety[i];
			if(x < w && y < h){
				dist = distance(iData, anchorX, anchorY, x, y, vectorOffsetx, vectorOffsety, w, h);
				insert(dist, x, y, similarPixelsX, similarPixelsY, similarPixelsDistance, &numOfSimilarPixels);
			}
		}

		for(int i = 0; i < numOfPredictors; ++i){
			unsigned sum = 0;
			for(int j = 0; j < numOfSimilarPixels; ++j){
				int prediction = predicted[i][similarPixelsX[j] + similarPixelsY[j] * w ];
				int pixel = iData[ similarPixelsX[j] + similarPixelsY[j] * w ];
				sum += (prediction - pixel) * (prediction - pixel);
			}
			if(numOfSimilarPixels == 0){
				penalties[i] = 0;
			}else{
				penalties[i] = (float)sum / numOfSimilarPixels;
			}
		}

		float sum = 0;
		float penaltiesSum = 0;
		for(int i = 0; i < numOfPredictors; ++i){
			sum += predicted[i][absolutePosition] / penalties[i];
			penaltiesSum += 1 / penalties[i];
		}

		delete penalties;
		if(penaltiesSum == 0){
			return 0;
		}else{
			return sum / penaltiesSum;
		}
	}

	__global__ void predict(void* diData, void* doData, void** dPredicted, unsigned numOfPredictors, void* dPredictionError, void* dRadiusOffsetx, void* dRadiusOffsety, void* dVectorOffsetx, void* dVectorOffsety, unsigned w, unsigned h) {
		unsigned absolutePosition = threadIdx.x + blockIdx.x * THREADS;
		if(absolutePosition >= w*h){
			return;
		}

		byte* iData = (byte*) diData;
		byte* oData = (byte*) doData;
		byte** predicted = (byte**) dPredicted;
		int* predictionError = (int*) dPredictionError;

		unsigned* radiusOffsetx = (unsigned*) dRadiusOffsetx;
		unsigned* radiusOffsety = (unsigned*) dRadiusOffsety;
		unsigned* vectorOffsetx = (unsigned*) dVectorOffsetx;
		unsigned* vectorOffsety = (unsigned*) dVectorOffsety;

		int prediction = predict(iData, predicted, numOfPredictors, radiusOffsetx, radiusOffsety, vectorOffsetx, vectorOffsety, w, h);
		oData[absolutePosition] = prediction;
		predictionError[absolutePosition] = iData[absolutePosition] - prediction;
	}
}

PGMCBPCCUDA::PGMCBPCCUDA(PGMImage& input, PGMImage& output, PGMImage& outputError):locked(false), input(input), output(output), outputError(outputError), w(input.getWidth()), h(input.getHeight()), size(input.getSize()){
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
	CUDA_CHECK_RETURN(cudaFree((void*) dPredictionError));
	CUDA_CHECK_RETURN(cudaDeviceReset());
}

bool PGMCBPCCUDA::init(){
	if(locked){
		return false;
	}
	locked = true;

	iData = new byte[size];
	oData = new byte[size];
	input.readIntoArray(iData, 0, size);
	predictionError = new int[size];

	CUDA_CHECK_RETURN(cudaMalloc(&diData, sizeof(byte) * size));
	CUDA_CHECK_RETURN(cudaMalloc(&doData, sizeof(byte) * size));
	CUDA_CHECK_RETURN(cudaMemcpy(diData, iData, sizeof(byte) * size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMalloc(&dPredictionError, sizeof(int) * size));

	CUDA_CHECK_RETURN(cudaMalloc(&dRadiusOffsetx, sizeof(radiusOffsetx)));
	CUDA_CHECK_RETURN(cudaMalloc(&dRadiusOffsety, sizeof(radiusOffsety)));
	CUDA_CHECK_RETURN(cudaMemcpy(dRadiusOffsetx, radiusOffsetx, sizeof(radiusOffsetx), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(dRadiusOffsety, radiusOffsety, sizeof(radiusOffsety), cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc(&dVectorOffsetx, sizeof(vectorOffsetx)));
	CUDA_CHECK_RETURN(cudaMalloc(&dVectorOffsety, sizeof(vectorOffsety)));
	CUDA_CHECK_RETURN(cudaMemcpy(dVectorOffsetx, vectorOffsetx, sizeof(vectorOffsetx), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(dVectorOffsety, vectorOffsety, sizeof(vectorOffsety), cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc(&dPredicted, sizeof(void*) * predictors.size()));
	void** hPredicted = new void*[predictors.size()];
	for(int i = 0; i < predictors.size(); ++i){
		CUDA_CHECK_RETURN(cudaMalloc(&hPredicted[i], sizeof(byte) * size));
		predictors[i]->predict(diData, hPredicted[i], w, h);
	}
	CUDA_CHECK_RETURN(cudaMemcpy(dPredicted, hPredicted, sizeof(void*) * predictors.size(), cudaMemcpyHostToDevice));
	delete hPredicted;

	return true;
}

bool PGMCBPCCUDA::addPredictor(Predictor* predictor){
	if(locked){
		return false;
	}
	predictors.push_back(predictor);
	return true;
}

void PGMCBPCCUDA::predict(){
	#ifdef DEBUG
	::predict<<<1, THREADS>>>(diData, doData, dPredicted, predictors.size(), dPredictionError, dRadiusOffsetx, dRadiusOffsety, dVectorOffsetx, dVectorOffsety, w, h);
	#else
	::predict<<<size/THREADS + 1, THREADS>>>(diData, doData, dPredicted, predictors.size(), dPredictionError, dRadiusOffsetx, dRadiusOffsety, dVectorOffsetx, dVectorOffsety, w, h);
	#endif
	CUDA_CHECK_RETURN(cudaMemcpy(oData, doData, sizeof(byte) * size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(predictionError, dPredictionError, sizeof(int) * size, cudaMemcpyDeviceToHost));

	for(unsigned j = 0; j < h; ++j){
		for(unsigned i = 0; i < w; ++i){
			output.writePixel(i, j, oData[i + j*w]);
		}
	}
	for(unsigned j = 0; j < h; ++j){
		for(unsigned i = 0; i < w; ++i){
			int error = predictionError[i + j*w];
			if(error < 0){
				error = -error;
			}
			outputError.writePixel(i, j, error);
		}
	}
}

bool PGMCBPCCUDA::getStaticPrediction(unsigned i){
	if(i >= predictors.size()){
		return false;
	}
	CUDA_CHECK_RETURN(cudaMemcpy(oData, dPredicted[i], sizeof(byte) * size, cudaMemcpyDeviceToHost));

	for(unsigned j = 0; j < h; ++j){
		for(unsigned i = 0; i < w; ++i){
			output.writePixel(i, j, oData[i + j*w]);
		}
	}
	return true;
}
