#include "PGMCBPCCUDA.h"
#include <iostream>

#define SWAP(a, b) (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))

struct PixelDistance{
	unsigned x, y, distance;
	__device__ PixelDistance(unsigned x, unsigned y, unsigned distance) : x(x), y(y), distance(distance){}
	__device__ PixelDistance() : x(0), y(0), distance(0){}
};

namespace {
	__device__ void insert(PixelDistance pixelDist, PixelDistance similarPixels[M], unsigned* numOfSimilarPixels){
		for(int i = 0; i < *numOfSimilarPixels; ++i){
			if(pixelDist.distance < similarPixels[i].distance){
				SWAP(pixelDist.distance, similarPixels[i].distance);
				SWAP(pixelDist.x, similarPixels[i].x);
				SWAP(pixelDist.y, similarPixels[i].y);
			}
		}

		if(*numOfSimilarPixels < M){
			similarPixels[*numOfSimilarPixels] = pixelDist;
			++*numOfSimilarPixels;
		}
	}

	__device__ unsigned distance(	byte* iData,
									unsigned anchorX,
									unsigned anchorY,
									unsigned x,
									unsigned y,
									PixelOffset* vectorOffset,
									unsigned w,
									unsigned h)
	{
		unsigned x1, x2, y1, y2;
		int sum = 0, pix1, pix2;
		for(int i = 0; i < D; ++i){
			x1 = anchorX + vectorOffset[i].x;
			y1 = anchorY + vectorOffset[i].y;
			x2 = x + vectorOffset[i].x;
			y2 = y + vectorOffset[i].y;
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

	__device__ byte predict(byte* iData,
							byte** predicted,
							unsigned numOfPredictors,
							PixelOffset* radiusOffset,
							PixelOffset* vectorOffset,
							unsigned w,
							unsigned h)
	{
		unsigned absolutePosition = threadIdx.x + blockIdx.x * THREADS;
		unsigned anchorX = absolutePosition % w;
		unsigned anchorY = absolutePosition / w;
		unsigned numOfSimilarPixels = 0;
		PixelDistance similarPixels[M];
		PixelDistance pixelDist;
		if(numOfPredictors == 0){
			return 0;
		}

		for(int i = 0; i < R_A; ++i){
			pixelDist.x = anchorX + radiusOffset[i].x;
			pixelDist.y = anchorY + radiusOffset[i].y;
			if(pixelDist.x < w && pixelDist.y < h){
				pixelDist.distance = distance(iData, anchorX, anchorY, pixelDist.x, pixelDist.y, vectorOffset, w, h);
				insert(pixelDist, similarPixels, &numOfSimilarPixels);
			}
		}
		if(numOfSimilarPixels == 0){
			return 0;
		}

		float* penalties = new float[numOfPredictors];
		for(int i = 0; i < numOfPredictors; ++i){
			unsigned sum = 0;
			for(int j = 0; j < numOfSimilarPixels; ++j){
				int prediction = predicted[i][similarPixels[j].x + similarPixels[j].y * w ];
				int pixel = iData[ similarPixels[j].x + similarPixels[j].y * w ];
				sum += (prediction - pixel) * (prediction - pixel);
			}
			if(sum == 0){
				delete[] penalties;
				return predicted[i][absolutePosition];
			}
			penalties[i] = (float)sum / numOfSimilarPixels;
		}

		float sum = 0;
		float penaltiesSum = 0;
		for(int i = 0; i < numOfPredictors; ++i){
			sum += predicted[i][absolutePosition] / penalties[i];
			penaltiesSum += 1 / penalties[i];
		}
		delete[] penalties;
		return sum / penaltiesSum;
	}

	__global__ void predict(void* diData,
							void* dpData,
							void** dPredicted,
							unsigned numOfPredictors,
							void* dRadiusOffset,
							void* dVectorOffset,
							ImageWHSize imageMeta)
	{
		unsigned absolutePosition = threadIdx.x + blockIdx.x * THREADS;
		if(absolutePosition >= imageMeta.size){
			return;
		}

		byte* iData = (byte*) diData;
		byte* pData = (byte*) dpData;
		byte** predicted = (byte**) dPredicted;
		PixelOffset* radiusOffset = (PixelOffset*) dRadiusOffset;
		PixelOffset* vectorOffset = (PixelOffset*) dVectorOffset;

		byte prediction = predict(iData, predicted, numOfPredictors, radiusOffset, vectorOffset, imageMeta.w, imageMeta.h);
		pData[absolutePosition] = prediction;
	}

	__global__ void errorCorrect(	void* diData,
									void* dpData,
									void* doData,
									void* deData,
									void* dRadiusOffset,
									void* dVectorOffset,
									ImageWHSize imageMeta)
	{
		unsigned absolutePosition = threadIdx.x + blockIdx.x * THREADS;
		unsigned anchorX = absolutePosition % imageMeta.w;
		unsigned anchorY = absolutePosition / imageMeta.w;
		if(anchorX >= imageMeta.w || anchorY >= imageMeta.h){
			return;
		}
		unsigned numOfSimilarPixels = 0;
		PixelDistance similarPixels[M];
		PixelDistance pixelDist;

		byte* iData = (byte*) diData;
		byte* pData = (byte*) dpData;
		byte* oData = (byte*) doData;
		short* eData = (short*) deData;
		PixelOffset* radiusOffset = (PixelOffset*) dRadiusOffset;
		PixelOffset* vectorOffset = (PixelOffset*) dVectorOffset;

		for(int i = 0; i < R_A; ++i){
			pixelDist.x = anchorX + radiusOffset[i].x;
			pixelDist.y = anchorY + radiusOffset[i].y;
			if(pixelDist.x < imageMeta.w && pixelDist.y < imageMeta.h){
				pixelDist.distance = distance(iData, anchorX, anchorY, pixelDist.x, pixelDist.y, vectorOffset, imageMeta.w, imageMeta.h);
				insert(pixelDist, similarPixels, &numOfSimilarPixels);
			}
		}
		if(numOfSimilarPixels == 0){
			oData[absolutePosition] = pData[absolutePosition];
			eData[absolutePosition] = iData[absolutePosition] - pData[absolutePosition];
		}

		int errorSum = 0;
		for(int i = 0; i < numOfSimilarPixels; ++i){
			errorSum += pData[ similarPixels[i].x + similarPixels[i].y * imageMeta.w ] - iData[ similarPixels[i].x + similarPixels[i].y * imageMeta.w ];
		}
		int prediction = (int)pData[absolutePosition] + errorSum / (int)numOfSimilarPixels;
		if(prediction < 0){
			prediction = 0;
		}else if(prediction > 255){
			prediction = 255;
		}
		oData[absolutePosition] = prediction;
		eData[absolutePosition] = iData[absolutePosition] - prediction;
	}
}

PGMCBPCCUDA::PGMCBPCCUDA(	vector<PGMImage>& inputImages,
							vector<PGMImage>& outputImages,
							vector<PGMImage>& errorImages,
							vector<Predictor*>& predictors
						):
		inputImages(inputImages),
		outputImages(outputImages),
		errorImages(errorImages),
		predictors(predictors)
{
	for(auto& inputImage : inputImages){
		imagesMeta.emplace_back(inputImage.getWidth(), inputImage.getHeight(), inputImage.getSize());
		iData.push_back(inputImage.getBuffer());
	}
	for(auto& outputImage : outputImages){
		oData.push_back(outputImage.getBuffer());
	}
	for(auto& imageMeta : imagesMeta){
		eData.push_back(new short[imageMeta.size]);
	}

	for(int i = 0; i < R_A; ++i){
		radiusOffset[i].x = i % (2*R + 1) - R;
		radiusOffset[i].y = i / (2*R + 1) - R;
	}
	for(int i = 0; i < D; ++i){
		vectorOffset[i].x = i % (D_R + 2) - D_R;
		vectorOffset[i].y = i / (D_R + 2) - D_R;
	}

	for(int i = 0; i < iData.size(); ++i){
		cout<<"Static prediction "<<i+1<<"/"<<iData.size()<<endl;
		diData.push_back(NULL);
		dpData.push_back(NULL);
		doData.push_back(NULL);
		deData.push_back(NULL);
		dPredicted.push_back(NULL);

		CUDA_CHECK_RETURN(cudaMalloc(&diData[i], sizeof(byte) * imagesMeta[i].size));
		CUDA_CHECK_RETURN(cudaMalloc(&dpData[i], sizeof(byte) * imagesMeta[i].size));
		CUDA_CHECK_RETURN(cudaMalloc(&doData[i], sizeof(byte) * imagesMeta[i].size));
		CUDA_CHECK_RETURN(cudaMalloc(&deData[i], sizeof(short) * imagesMeta[i].size));
		CUDA_CHECK_RETURN(cudaMemcpy(diData[i], iData[i], sizeof(byte) * imagesMeta[i].size, cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMalloc(&dPredicted[i], sizeof(void*) * predictors.size()));
		vector<void*> hPredicted(predictors.size());
		for(int j = 0; j < predictors.size(); ++j){
			CUDA_CHECK_RETURN(cudaMalloc(&hPredicted[j], sizeof(byte) * imagesMeta[i].size));
			predictors[j]->predict(diData[i], hPredicted[j], imagesMeta[i].w, imagesMeta[i].h);
		}
		CUDA_CHECK_RETURN(cudaMemcpy(dPredicted[i], hPredicted.data(), sizeof(void*) * predictors.size(), cudaMemcpyHostToDevice));
		cout<<"DONE"<<endl;
	}

	CUDA_CHECK_RETURN(cudaMalloc(&dRadiusOffset, sizeof(radiusOffset)));
	CUDA_CHECK_RETURN(cudaMemcpy(dRadiusOffset, radiusOffset, sizeof(radiusOffset), cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc(&dVectorOffset, sizeof(vectorOffset)));
	CUDA_CHECK_RETURN(cudaMemcpy(dVectorOffset, vectorOffset, sizeof(vectorOffset), cudaMemcpyHostToDevice));
}

PGMCBPCCUDA::~PGMCBPCCUDA(){
	CUDA_CHECK_RETURN(cudaDeviceReset());
	for(auto p : eData){
		delete[] p;
	}
}

void PGMCBPCCUDA::predict(){
	for(int i = 0; i < inputImages.size(); ++i){
		cout<<"Prediction "<<i+1<<"/"<<inputImages.size()<<endl;
		::predict<<<imagesMeta[i].size/THREADS + 1, THREADS>>>(		diData[i],
																	dpData[i],
																	dPredicted[i],
																	predictors.size(),
																	dRadiusOffset,
																	dVectorOffset,
																	imagesMeta[i]
																);
		::errorCorrect<<<imagesMeta[i].size/THREADS + 1, THREADS>>>(diData[i],
																	dpData[i],
																	doData[i],
																	deData[i],
																	dRadiusOffset,
																	dVectorOffset,
																	imagesMeta[i]
																);

		CUDA_CHECK_RETURN(cudaMemcpy(oData[i], doData[i], sizeof(byte) * imagesMeta[i].size, cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpy(eData[i], deData[i], sizeof(short) * imagesMeta[i].size, cudaMemcpyDeviceToHost));

		for(unsigned k = 0; k < imagesMeta[i].h; ++k){
			for(unsigned j = 0; j < imagesMeta[i].w; ++j){
				short error = eData[i][j + k*imagesMeta[i].w];
				if(error < 0){
					error = -error;
				}
				errorImages[i].writePixel(j, k, error);
			}
		}
		cout<<"DONE"<<endl;
	}
}

bool PGMCBPCCUDA::getStaticPrediction(unsigned predictorIndex){
	if(predictorIndex >= predictors.size()){
		return false;
	}

	for(int i = 0; i < inputImages.size(); ++i){
		void** hPredicted = new void*[predictors.size()];
		CUDA_CHECK_RETURN(cudaMemcpy(hPredicted, dPredicted[i], sizeof(void*) * predictors.size(), cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpy(oData[i], hPredicted[predictorIndex], sizeof(byte) * imagesMeta[i].size, cudaMemcpyDeviceToHost));
		delete[] hPredicted;
	}

	return true;
}
