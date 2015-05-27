#ifndef PGMCBPD_H_
#define PGMCBPD_H_

#include <vector>

#include "../PGM/PGMImage.h"
#include "../PGM/PGMImageError.h"
#include "../staticPredictors/Predictor.h"
#include "../config.h"
#include "../util.h"

struct PixelDistance{
	unsigned x, y, distance;
	PixelDistance(unsigned x, unsigned y, unsigned distance) : x(x), y(y), distance(distance){}
	PixelDistance() : x(0), y(0), distance(0){}
};

struct PGMCBPD{
	PGMCBPD(vector<PGMImageError>& inputImagesError,
			vector<PGMImage>& outputImages,
			vector<Predictor*>& predictors);
	~PGMCBPD();

	void decode();

private:
	void decode(unsigned imageIndex);
	byte predictElement(unsigned imageIndex, unsigned anchorX, unsigned anchorY);
	int errorCorrectElement(unsigned imageIndex, unsigned anchorX, unsigned anchorY);

	unsigned distance(unsigned imageIndex, unsigned anchorX, unsigned anchorY, unsigned x, unsigned y);
	void insert(PixelDistance pixelDist, PixelDistance similarPixels[M], unsigned* numOfSimilarPixels);

	vector<PGMImageError>& inputImagesError;
	vector<PGMImage>& outputImages;
	vector<Predictor*>& predictors;

	vector<ImageWHSize> imagesMeta;
	vector<short*> iData;
	vector<byte*> oData;
	vector<byte*> pMemoData;
	vector<byte*> pData;

	PixelOffset radiusOffset[R_A];
	PixelOffset vectorOffset[D];
};

#endif /* PGMCBPD_H_ */
