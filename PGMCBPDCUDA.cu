#include <iostream>
#include "PGMCBPDCUDA.h"
#include "decoderPredictors/predictors.h"

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

	__device__ byte predict(byte* oData,
							unsigned anchorX,
							unsigned anchorY,
							unsigned w,
							unsigned h,
							PixelOffset* radiusOffset,
							PixelOffset* vectorOffset)
	{
		unsigned numOfSimilarPixels = 0;
		PixelDistance similarPixels[M];
		PixelDistance pixelDist;

		for(int i = 0; i < R_A; ++i){
			pixelDist.x = anchorX + radiusOffset[i].x;
			pixelDist.y = anchorY + radiusOffset[i].y;
			if(pixelDist.x < w && pixelDist.y < h){
				pixelDist.distance = distance(oData, anchorX, anchorY, pixelDist.x, pixelDist.y, vectorOffset, w, h);
				insert(pixelDist, similarPixels, &numOfSimilarPixels);
			}
		}
		if(numOfSimilarPixels == 0){
			return 0;
		}

		byte* predictions = new byte[numOfPredictors];
		float* penalties = new float[numOfPredictors];
		for(int i = 0; i < numOfPredictors; ++i){
			unsigned sum = 0;
			for(int j = 0; j < numOfSimilarPixels; ++j){
				predictions[i] = predictors[i](oData, similarPixels[j].x, similarPixels[j].y, w, h);
				int pixel = oData[ similarPixels[j].x + similarPixels[j].y * w ];
				sum += (predictions[i] - pixel) * (predictions[i] - pixel);
			}
			if(sum == 0){
				delete[] predictions;
				delete[] penalties;
				return predictions[i];
			}
			penalties[i] = (float)sum / numOfSimilarPixels;
		}

		float sum = 0;
		float penaltiesSum = 0;
		for(int i = 0; i < numOfPredictors; ++i){
			sum += predictions[i] / penalties[i];
			penaltiesSum += 1 / penalties[i];
		}
		delete[] predictions;
		delete[] penalties;
		return sum / penaltiesSum;
	}

	__device__ int errorCorrect(	void* doData,
									void* dpData,
									unsigned anchorX,
									unsigned anchorY,
									unsigned w,
									unsigned h,
									void* dRadiusOffset,
									void* dVectorOffset)
	{
		unsigned numOfSimilarPixels = 0;
		PixelDistance similarPixels[M];
		PixelDistance pixelDist;

		byte* oData = (byte*) doData;
		byte* pData = (byte*) dpData;
		PixelOffset* radiusOffset = (PixelOffset*) dRadiusOffset;
		PixelOffset* vectorOffset = (PixelOffset*) dVectorOffset;

		for(int i = 0; i < R_A; ++i){
			pixelDist.x = anchorX + radiusOffset[i].x;
			pixelDist.y = anchorY + radiusOffset[i].y;
			if(pixelDist.x < w && pixelDist.y < h){
				pixelDist.distance = distance(oData, anchorX, anchorY, pixelDist.x, pixelDist.y, vectorOffset, w, h);
				insert(pixelDist, similarPixels, &numOfSimilarPixels);
			}
		}
		if(numOfSimilarPixels == 0){
			return 0;
		}

		int errorSum = 0;
		for(int i = 0; i < numOfSimilarPixels; ++i){
			errorSum += pData[ similarPixels[i].x + similarPixels[i].y * w ] - oData[ similarPixels[i].x + similarPixels[i].y * w ];
		}
		return errorSum / (int)numOfSimilarPixels;
	}

	__global__ void decode(	void* diData,
							void* doData,
							void* dpData,
							void* dRadiusOffset,
							void* dVectorOffset,
							ImageWHSize imageMeta)
	{
		short* iData = (short*) diData;
		byte* oData = (byte*) doData;
		byte* pData = (byte*) dpData;
		PixelOffset* radiusOffset = (PixelOffset*) dRadiusOffset;
		PixelOffset* vectorOffset = (PixelOffset*) dVectorOffset;

		for(unsigned y = 0; y < imageMeta.h; ++y){
			for(unsigned x = 0; x < imageMeta.w; ++x){
				unsigned pos = x + y * imageMeta.w;
				int prediction = predict(oData, x, y, imageMeta.w, imageMeta.h, radiusOffset, vectorOffset);
				prediction += errorCorrect(oData, pData, x, y, imageMeta.w, imageMeta.h, radiusOffset, vectorOffset);
				if(prediction < 0){
					prediction = 0;
				}else if(prediction > 255){
					prediction = 255;
				}
				pData[pos] = prediction;
				oData[pos] = prediction + iData[pos];
			}
		}
	}
}

PGMCBPDCUDA::PGMCBPDCUDA(	vector<PGMImageError>& inputImagesError,
							vector<PGMImage>& outputImages,
							vector<PGMImage>& predictionImages
						):
		inputImagesError(inputImagesError),
		outputImages(outputImages),
		predictionImages(predictionImages)
{
	for(auto& inputImageError : inputImagesError){
		streams.emplace_back();
		cudaStreamCreate(&streams.back());
		imagesMeta.emplace_back(inputImageError.getWidth(), inputImageError.getHeight(), inputImageError.getSize());
		iData.push_back(inputImageError.getBuffer());
	}
	for(auto& outputImage : outputImages){
		oData.push_back(outputImage.getBuffer());
	}
	for(auto& predictionImage : predictionImages){
		pData.push_back(predictionImage.getBuffer());
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
		cout<<"Memory allocation "<<i+1<<"/"<<iData.size()<<endl;

		diData.push_back(NULL);
		doData.push_back(NULL);
		dpData.push_back(NULL);

		CUDA_CHECK_RETURN(cudaMalloc(&diData[i], sizeof(short) * imagesMeta[i].size));
		CUDA_CHECK_RETURN(cudaMalloc(&doData[i], sizeof(byte) * imagesMeta[i].size));
		CUDA_CHECK_RETURN(cudaMalloc(&dpData[i], sizeof(byte) * imagesMeta[i].size));
		CUDA_CHECK_RETURN(cudaMemcpy(diData[i], iData[i], sizeof(short) * imagesMeta[i].size, cudaMemcpyHostToDevice));

		cout<<"DONE"<<endl;
	}

	CUDA_CHECK_RETURN(cudaMalloc(&dRadiusOffset, sizeof(radiusOffset)));
	CUDA_CHECK_RETURN(cudaMemcpy(dRadiusOffset, radiusOffset, sizeof(radiusOffset), cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc(&dVectorOffset, sizeof(vectorOffset)));
	CUDA_CHECK_RETURN(cudaMemcpy(dVectorOffset, vectorOffset, sizeof(vectorOffset), cudaMemcpyHostToDevice));
}

PGMCBPDCUDA::~PGMCBPDCUDA(){
	CUDA_CHECK_RETURN(cudaDeviceReset());
}

void PGMCBPDCUDA::decode(){
	for(int i = 0; i < streams.size(); ++i){
		::decode<<<1, 1, 0, streams[i]>>>(	diData[i],
											doData[i],
											dpData[i],
											dRadiusOffset,
											dVectorOffset,
											imagesMeta[i]);
		CUDA_CHECK_RETURN(cudaStreamDestroy(streams[i]));
	}

	for(int i = 0; i < streams.size(); ++i){
		CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[i]));
		CUDA_CHECK_RETURN(cudaMemcpy(oData[i], doData[i], sizeof(byte) * imagesMeta[i].size, cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpy(pData[i], dpData[i], sizeof(byte) * imagesMeta[i].size, cudaMemcpyDeviceToHost));
	}
}
