#include <iostream>

#include "PGMCBPC.h"
#include "../config.h"

#define SWAP(a, b) (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))

PGMCBPC::PGMCBPC(	vector<PGMImage>& inputImages,
					vector<PGMImage>& outputImages,
					vector<PGMImageError>& errorImages,
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
		pData.push_back(new byte[predictors.size()]);
	}
	for(auto& errorImage : errorImages){
		eData.push_back(errorImage.getBuffer());
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
		for(int j = 0; j < predictors.size(); ++j){
			pData[i][j] = new byte[imagesMeta[i].size];
			predictors[j]->predictAll(iData[i], pData[i][j], imagesMeta[i].w, imagesMeta[i].h);
		}
		cout<<"DONE"<<endl;
	}
}

PGMCBPC::~PGMCBPC(){
	for(int i = 0; i < pData.size(); ++i){
		for(int j = 0; j < predictors.size(); ++j){
			delete[] pData[i][j];
		}
		delete[] pData[i];
	}
}

bool PGMCBPC::getStaticPrediction(unsigned predictorIndex){
	if(predictorIndex >= predictors.size()){
		return false;
	}

	for(unsigned i = 0; i < inputImages.size(); ++i){
		inputImages[i].setBuffer(pData[i][predictorIndex]);
		delete[] iData[i];
	}

	return true;
}

void PGMCBPC::predict(){
	for(unsigned i = 0; i < inputImages.size(); ++i){
		cout<<"Prediction "<<i+1<<"/"<<inputImages.size()<<endl;
		predict(i);
		errorCorrect(i);

		cout<<"DONE"<<endl;
	}
}

void PGMCBPC::predict(unsigned imageIndex){
	if(imageIndex > inputImages.size()){
		return;
	}

	for(unsigned y = 0; y < imagesMeta[imageIndex].h; ++y){
		for(unsigned x = 0; x < imagesMeta[imageIndex].w; ++ x){
			byte prediction = predictElement(imageIndex, x, y);
			oData[imageIndex][y * imagesMeta[imageIndex].w + x] = prediction;
		}
	}
}

byte PGMCBPC::predictElement(unsigned imageIndex, unsigned anchorX, unsigned anchorY){
	unsigned numOfSimilarPixels = 0;
	PixelDistance similarPixels[M];
	PixelDistance pixelDist;
	if(predictors.size() == 0){
		return 0;
	}

	for(unsigned i = 0; i < R_A; ++i){
		pixelDist.x = anchorX + radiusOffset[i].x;
		pixelDist.y = anchorY + radiusOffset[i].y;
		if(pixelDist.x < w && pixelDist.y < h){
			pixelDist.distance = distance(iData[imageIndex], anchorX, anchorY, pixelDist.x, pixelDist.y);
			insert(pixelDist, similarPixels, &numOfSimilarPixels);
		}
	}
	if(numOfSimilarPixels == 0){
		return 0;
	}

	float* penalties = new float[numOfPredictors];
	for(int i = 0; i < numOfPredictors; ++i){
		unsigned sum = 0;
		int staticPrediction;
		for(int j = 0; j < numOfSimilarPixels; ++j){
			staticPrediction = predicted[i][similarPixels[j].x + similarPixels[j].y * w ];
			int pixel = iData[imageIndex][ similarPixels[j].x + similarPixels[j].y * w ];
			sum += (staticPrediction - pixel) * (staticPrediction - pixel);
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

unsigned PGMCBPC::distance(unsigned imageIndex, unsigned anchorX, unsigned anchorY, unsigned x, unsigned y){
	unsigned x1, x2, y1, y2;
	int sum = 0, pix1, pix2;
	for(int i = 0; i < D; ++i){
		x1 = anchorX + vectorOffset[i].x;
		y1 = anchorY + vectorOffset[i].y;
		x2 = x + vectorOffset[i].x;
		y2 = y + vectorOffset[i].y;
		pix1 = pix2 = 0;
		if(x1 < imagesMeta[imageIndex].w && y1 < imagesMeta[imageIndex].h){
			pix1 = iData[x1 + y1 * w];
		}
		if(x2 < imagesMeta[imageIndex].w && y2 < imagesMeta[imageIndex].h){
			pix2 = iData[x2 + y2 * w];
		}
		sum += (pix1-pix2) * (pix1-pix2);
	}
	return sum;
}

void PGMCBPC::insert(PixelDistance pixelDist, PixelDistance similarPixels[M], unsigned* numOfSimilarPixels){
	for(unsigned i = 0; i < *numOfSimilarPixels; ++i){
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

void PGMCBPC::errorCorrect(unsigned imageIndex){
	unsigned absolutePosition = threadIdx.x + blockIdx.x * THREADS;
	unsigned anchorX = absolutePosition % imageMeta.w;
	unsigned anchorY = absolutePosition / imageMeta.w;
	if(anchorX >= imageMeta.w || anchorY >= imageMeta.h){
		return;
	}
	unsigned numOfSimilarPixels = 0;
	PixelDistance similarPixels[M];
	PixelDistance pixelDist;

	for(int i = 0; i < R_A; ++i){
		pixelDist.x = anchorX + radiusOffset[i].x;
		pixelDist.y = anchorY + radiusOffset[i].y;
		if(pixelDist.x < imageMeta.w && pixelDist.y < imageMeta.h){
			pixelDist.distance = distance(imageIndex, anchorX, anchorY, pixelDist.x, pixelDist.y);
			insert(pixelDist, similarPixels, &numOfSimilarPixels);
		}
	}
	if(numOfSimilarPixels == 0){
		oData[imageIndex][absolutePosition] = pData[imageIndex][absolutePosition];
		eData[imageIndex][absolutePosition] = iData[imageIndex][absolutePosition] - pData[imageIndex][absolutePosition];
		return;
	}

	int errorSum = 0;
	for(int i = 0; i < numOfSimilarPixels; ++i){
		errorSum += pData[imageIndex][ similarPixels[i].x + similarPixels[i].y * imageMeta.w ] - iData[imageIndex][ similarPixels[i].x + similarPixels[i].y * imageMeta.w ];
	}
	int prediction = (int)pData[imageIndex][absolutePosition] + errorSum / (int)numOfSimilarPixels;
	if(prediction < 0){
		prediction = 0;
	}else if(prediction > 255){
		prediction = 255;
	}
	oData[imageIndex][absolutePosition] = prediction;
	eData[imageIndex][absolutePosition] = iData[imageIndex][absolutePosition] - prediction;
}
