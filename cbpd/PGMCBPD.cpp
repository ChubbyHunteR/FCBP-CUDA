#include <iostream>
#include <cstring>

#include "PGMCBPD.h"

#define SWAP(a, b) (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))

void PGMCBPD::insert(PixelDistance pixelDist, PixelDistance similarPixels[M], unsigned* numOfSimilarPixels){
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

unsigned PGMCBPD::distance(unsigned imageIndex, unsigned anchorX, unsigned anchorY, unsigned x, unsigned y){
	unsigned x1, x2, y1, y2;
	int sum = 0, pix1, pix2;
	for(int i = 0; i < D; ++i){
		x1 = anchorX + vectorOffset[i].x;
		y1 = anchorY + vectorOffset[i].y;
		x2 = x + vectorOffset[i].x;
		y2 = y + vectorOffset[i].y;
		pix1 = pix2 = 0;
		if(x1 < imagesMeta[imageIndex].w && y1 < imagesMeta[imageIndex].h){
			pix1 = iData[imageIndex][x1 + y1 * imagesMeta[imageIndex].w];
		}
		if(x2 < imagesMeta[imageIndex].w && y2 < imagesMeta[imageIndex].h){
			pix2 = iData[imageIndex][x2 + y2 * imagesMeta[imageIndex].w];
		}
		sum += (pix1-pix2) * (pix1-pix2);
	}
	return sum;
}

byte PGMCBPD::predictElement(unsigned imageIndex, unsigned anchorX, unsigned anchorY){
	unsigned numOfSimilarPixels = 0;
	PixelDistance similarPixels[M];
	PixelDistance pixelDist;

	for(int i = 0; i < R_A; ++i){
		pixelDist.x = anchorX + radiusOffset[i].x;
		pixelDist.y = anchorY + radiusOffset[i].y;
		if(pixelDist.x < imagesMeta[imageIndex].w && pixelDist.y < imagesMeta[imageIndex].h){
			pixelDist.distance = distance(imageIndex, anchorX, anchorY, pixelDist.x, pixelDist.y);
			insert(pixelDist, similarPixels, &numOfSimilarPixels);
		}
	}
	if(numOfSimilarPixels == 0){
		return 0;
	}

	float* penalties = new float[predictors.size()];
	for(int i = 0; i < predictors.size(); ++i){
		unsigned sum = 0;
		for(int j = 0; j < numOfSimilarPixels; ++j){
			int staticPrediction = pMemoData[imageIndex][i * imagesMeta[imageIndex].size + similarPixels[j].x + similarPixels[j].y * imagesMeta[imageIndex].w];
			int pixel = oData[imageIndex][ similarPixels[j].x + similarPixels[j].y * imagesMeta[imageIndex].w ];
			sum += (staticPrediction - pixel) * (staticPrediction - pixel);
		}
		if(sum == 0){
			return pMemoData[imageIndex][i * imagesMeta[imageIndex].size + anchorX + anchorY * imagesMeta[imageIndex].w];
		}
		penalties[i] = (float)sum / numOfSimilarPixels;
	}

	float sum = 0;
	float penaltiesSum = 0;
	for(int i = 0; i < predictors.size(); ++i){
		int prediction = pMemoData[imageIndex][i * imagesMeta[imageIndex].size + anchorX + anchorY * imagesMeta[imageIndex].w];
		sum += prediction / penalties[i];
		penaltiesSum += 1 / penalties[i];
	}
	delete[] penalties;
	return sum / penaltiesSum;
}

int PGMCBPD::errorCorrectElement(unsigned imageIndex, unsigned anchorX,	unsigned anchorY){
	unsigned numOfSimilarPixels = 0;
	PixelDistance similarPixels[M];
	PixelDistance pixelDist;

	for(int i = 0; i < R_A; ++i){
		pixelDist.x = anchorX + radiusOffset[i].x;
		pixelDist.y = anchorY + radiusOffset[i].y;
		if(pixelDist.x < imagesMeta[imageIndex].w && pixelDist.y < imagesMeta[imageIndex].h){
			pixelDist.distance = distance(imageIndex, anchorX, anchorY, pixelDist.x, pixelDist.y);
			insert(pixelDist, similarPixels, &numOfSimilarPixels);
		}
	}
	if(numOfSimilarPixels == 0){
		return 0;
	}

	int errorSum = 0;
	for(int i = 0; i < numOfSimilarPixels; ++i){
		errorSum += pData[imageIndex][ similarPixels[i].x + similarPixels[i].y * imagesMeta[imageIndex].w ] - oData[imageIndex][ similarPixels[i].x + similarPixels[i].y * imagesMeta[imageIndex].w ];
	}
	return errorSum / (int)numOfSimilarPixels;
}

void PGMCBPD::decode(unsigned imageIndex){
	for(unsigned y = 0; y < imagesMeta[imageIndex].h; ++y){
		for(unsigned x = 0; x < imagesMeta[imageIndex].w; ++x){
			unsigned pos = x + y * imagesMeta[imageIndex].w;
			for(unsigned i = 0; i < predictors.size(); ++i){
				pMemoData[imageIndex][i * imagesMeta[imageIndex].size + pos] = predictors[i]->predict(oData[imageIndex], x, y, imagesMeta[imageIndex].w, imagesMeta[imageIndex].h);
			}
			int prediction = pData[imageIndex][pos] = predictElement(imageIndex, x, y);
			prediction += errorCorrectElement(imageIndex, x, y);
			if(prediction < 0){
				prediction = 0;
			}else if(prediction > 255){
				prediction = 255;
			}
			oData[imageIndex][pos] = prediction + iData[imageIndex][pos];
		}
	}
}

PGMCBPD::PGMCBPD(	vector<PGMImageError>& inputImagesError,
					vector<PGMImage>& outputImages,
					vector<Predictor*>& predictors
				):
		inputImagesError(inputImagesError),
		outputImages(outputImages),
		predictors(predictors)
{
	for(auto& inputImageError : inputImagesError){
		imagesMeta.emplace_back(inputImageError.getWidth(), inputImageError.getHeight(), inputImageError.getSize());
		iData.push_back(inputImageError.getBuffer());
		pMemoData.push_back(new byte[predictors.size() * inputImageError.getSize()]);
		pData.push_back(new byte[inputImageError.getSize()]);
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
}

PGMCBPD::~PGMCBPD(){
	for(auto data : pMemoData){
		delete[] data;
	}
	for(auto data : pData){
		delete[] data;
	}
}

void PGMCBPD::decode(){
	for(unsigned i = 0; i < inputImagesError.size(); ++i){
		cout << "Waiting for " << i+1 << "/" << inputImagesError.size() << endl;
		decode(i);
		cout << "DONE" << endl;
	}
}
