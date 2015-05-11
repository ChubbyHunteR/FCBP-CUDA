#include <iostream>

#include "PGMCBPDCUDA.h"

#define SWAP(a, b) (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))
#define NUM_PREDICTORS 7

struct PixelDistance{
	unsigned x, y, distance;
	__device__ PixelDistance(unsigned x, unsigned y, unsigned distance) : x(x), y(y), distance(distance){}
	__device__ PixelDistance() : x(0), y(0), distance(0){}
};

namespace {
	__device__ byte predictGN(byte *oData, unsigned x, unsigned y, unsigned w, unsigned h) {
		y -= 2;
		int sum = 0;
		if(x < w && y < h){
			sum -= oData[y * w + x];
		}
		++y;
		if(x < w && y < h){
			sum += 2 * oData[y * w + x];
		}

		if(sum < 0){
			sum = 0;
		}else if(sum > 255){
			sum = 255;
		}
		return sum;
	}

	__device__ byte predictGW(byte *oData, unsigned x, unsigned y, unsigned w, unsigned h) {
		x -= 2;
		int sum = 0;
		if(x < w && y < h){
			sum -= oData[y * w + x];
		}
		++x;
		if(x < w && y < h){
			sum += 2 * oData[y * w + x];
		}

		if(sum < 0){
			sum = 0;
		}else if(sum > 255){
			sum = 255;
		}
		return sum;
	}

	__device__ byte predictN(byte *oData, unsigned x, unsigned y, unsigned w, unsigned h) {
		--y;
		if(x < w && y < h){
			return oData[y * w + x];
		}
		return 0;
	}

	__device__ byte predictNE(byte *oData, unsigned x, unsigned y, unsigned w, unsigned h) {
		++x;
		--y;
		if(x < w && y < h){
			return oData[y * w + x];
		}
		return 0;
	}

	__device__ byte predictNW(byte *oData, unsigned x, unsigned y, unsigned w, unsigned h) {
		--x;
		--y;
		if(x < w && y < h){
			return oData[y * w + x];
		}
		return 0;
	}

	__device__ byte predictPL(byte *oData, unsigned x, unsigned y, unsigned w, unsigned h) {
		--x;
		--y;
		int sum = 0;
		if(x < w && y < h){
			sum -= oData[y * w + x];
		}
		++x;
		if(x < w && y < h){
			sum += oData[y * w + x];
		}
		--x;
		++y;
		if(x < w && y < h){
			sum += oData[y * w + x];
		}

		if(sum < 0){
			sum = 0;
		}else if(sum > 255){
			sum = 255;
		}
		return sum;
	}

	__device__ byte predictW(byte *oData, unsigned x, unsigned y, unsigned w, unsigned h) {
		--x;
		if(x < w && y < h){
			return oData[y * w + x];
		}
		return 0;
	}

	__device__ byte (* predictors[])(byte *oData, unsigned x, unsigned y, unsigned w, unsigned h)
	{
		predictN,
		predictNW,
		predictGW,
		predictW,
		predictNE,
		predictGN,
		predictPL,
		NULL
	};

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
							short* pData,
							unsigned anchorX,
							unsigned anchorY,
							PixelOffset* radiusOffset,
							PixelOffset* vectorOffset,
							ImageWHSize imageMeta)
	{
		unsigned numOfSimilarPixels = 0;
		PixelDistance similarPixels[M];
		PixelDistance pixelDist;

		for(int i = 0; i < R_A; ++i){
			pixelDist.x = anchorX + radiusOffset[i].x;
			pixelDist.y = anchorY + radiusOffset[i].y;
			if(pixelDist.x < imageMeta.w && pixelDist.y < imageMeta.h){
				pixelDist.distance = distance(oData, anchorX, anchorY, pixelDist.x, pixelDist.y, vectorOffset, imageMeta.w, imageMeta.h);
				insert(pixelDist, similarPixels, &numOfSimilarPixels);
			}
		}
		if(numOfSimilarPixels == 0){
			return 0;
		}

		float penalties[NUM_PREDICTORS];
		for(int i = 0; i < NUM_PREDICTORS; ++i){
			unsigned sum = 0;
			int staticPrediction = -1;
			for(int j = 0; j < numOfSimilarPixels; ++j){
				staticPrediction = pData[i * imageMeta.size + similarPixels[j].x + similarPixels[j].y * imageMeta.w];
				if(staticPrediction == -1){
					pData[i * imageMeta.size + similarPixels[j].x + similarPixels[j].y * imageMeta.w] =
					staticPrediction =
					predictors[i](oData, similarPixels[j].x, similarPixels[j].y, imageMeta.w, imageMeta.h);
				}
				int pixel = oData[ similarPixels[j].x + similarPixels[j].y * imageMeta.w ];
				sum += (staticPrediction - pixel) * (staticPrediction - pixel);
			}
			if(sum == 0){
				return pData[i * imageMeta.size + anchorX + anchorY * imageMeta.w] =
						predictors[i](oData, anchorX, anchorY, imageMeta.w, imageMeta.h);
			}
			penalties[i] = (float)sum / numOfSimilarPixels;
		}

		float sum = 0;
		float penaltiesSum = 0;
		for(int i = 0; i < NUM_PREDICTORS; ++i){
			int prediction = pData[i * imageMeta.size + anchorX + anchorY * imageMeta.w] =
					predictors[i](oData, anchorX, anchorY, imageMeta.w, imageMeta.h);
			sum += prediction / penalties[i];
			penaltiesSum += 1 / penalties[i];
		}
		return sum / penaltiesSum;
	}

	__device__ int errorCorrect(	byte* oData,
									short* pData,
									byte* spData,
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
			errorSum += spData[ similarPixels[i].x + similarPixels[i].y * w ] - oData[ similarPixels[i].x + similarPixels[i].y * w ];
		}
		return errorSum / (int)numOfSimilarPixels;
	}

	__global__ void decode(	void* diData,
							void* doData,
							void* dpData,
							void* dspData,
							void* dRadiusOffset,
							void* dVectorOffset,
							ImageWHSize imageMeta)
	{
		short* iData = (short*) diData;
		byte* oData = (byte*) doData;
		short* pData = (short*) dpData;
		byte* spData = (byte*) dspData;
		PixelOffset* radiusOffset = (PixelOffset*) dRadiusOffset;
		PixelOffset* vectorOffset = (PixelOffset*) dVectorOffset;

		for(unsigned y = 0; y < imageMeta.h; ++y){
			for(unsigned x = 0; x < imageMeta.w; ++x){
				unsigned pos = x + y * imageMeta.w;
				int prediction = spData[pos] = predict(oData, pData, x, y, radiusOffset, vectorOffset, imageMeta);
				prediction += errorCorrect(oData, pData, spData, x, y, imageMeta.w, imageMeta.h, radiusOffset, vectorOffset);
				if(prediction < 0){
					prediction = 0;
				}else if(prediction > 255){
					prediction = 255;
				}
				oData[pos] = prediction + iData[pos];
			}
		}
	}
}

PGMCBPDCUDA::PGMCBPDCUDA(	vector<PGMImageError>& inputImagesError,
							vector<PGMImage>& outputImages
						):
		inputImagesError(inputImagesError),
		outputImages(outputImages)
{
	for(auto& inputImageError : inputImagesError){
		streams.emplace_back();
		cudaStreamCreate(&streams.back());
		imagesMeta.emplace_back(inputImageError.getWidth(), inputImageError.getHeight(), inputImageError.getSize());
		iData.push_back(inputImageError.getBuffer());
		pData.push_back(new short[NUM_PREDICTORS * inputImageError.getSize()]);
	}
	for(auto& outputImage : outputImages){
		oData.push_back(outputImage.getBuffer());
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

		diData.push_back(nullptr);
		doData.push_back(nullptr);
		dpData.push_back(nullptr);
		dspData.push_back(nullptr);

		CUDA_CHECK_RETURN(cudaMalloc(&diData[i], sizeof(short) * imagesMeta[i].size));
		CUDA_CHECK_RETURN(cudaMalloc(&doData[i], sizeof(byte) * imagesMeta[i].size));
		CUDA_CHECK_RETURN(cudaMalloc(&dpData[i], NUM_PREDICTORS * sizeof(short) * imagesMeta[i].size));
		CUDA_CHECK_RETURN(cudaMalloc(&dspData[i], sizeof(byte) * imagesMeta[i].size));
		CUDA_CHECK_RETURN(cudaMemset(dpData[i], -1, NUM_PREDICTORS * sizeof(short) * imagesMeta[i].size));
		CUDA_CHECK_RETURN(cudaMemcpy(diData[i], iData[i], sizeof(short) * imagesMeta[i].size, cudaMemcpyHostToDevice));

		cout<<"DONE"<<endl;
	}

	CUDA_CHECK_RETURN(cudaMalloc(&dRadiusOffset, sizeof(radiusOffset)));
	CUDA_CHECK_RETURN(cudaMemcpy(dRadiusOffset, radiusOffset, sizeof(radiusOffset), cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc(&dVectorOffset, sizeof(vectorOffset)));
	CUDA_CHECK_RETURN(cudaMemcpy(dVectorOffset, vectorOffset, sizeof(vectorOffset), cudaMemcpyHostToDevice));
}

PGMCBPDCUDA::~PGMCBPDCUDA(){
	for(auto& stream : streams){
		CUDA_CHECK_RETURN(cudaStreamDestroy(stream));
	}
	CUDA_CHECK_RETURN(cudaDeviceReset());
}

void PGMCBPDCUDA::decode(){
	for(int i = 0; i < streams.size(); ++i){
		::decode<<<1, 1, 0, streams[i]>>>(	diData[i],
											doData[i],
											dpData[i],
											dspData[i],
											dRadiusOffset,
											dVectorOffset,
											imagesMeta[i]);
	}

	for(int i = 0; i < streams.size(); ++i){
		cout << "Waiting for " << i+1 << "/" << streams.size() << endl;
		CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[i]));
		cout << "DONE" << endl;
	}

	for(int i = 0; i < streams.size(); ++i){
		cout << "Copying " << i+1 << "/" << streams.size() << endl;
		CUDA_CHECK_RETURN(cudaMemcpy(oData[i], doData[i], sizeof(byte) * imagesMeta[i].size, cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpy(pData[i], dpData[i], NUM_PREDICTORS * sizeof(short) * imagesMeta[i].size, cudaMemcpyDeviceToHost));
		cout << "DONE" << endl;
	}
}
