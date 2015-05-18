#ifndef PGMCBPC_H_
#define PGMCBPC_H_

#include <vector>

#include "../PGM/PGMImage.h"
#include "../PGM/PGMImageError.h"
#include "../staticPredictors/Predictor.h"
#include "../util.h"

struct PixelDistance{
	unsigned x, y, distance;
	PixelDistance(unsigned x, unsigned y, unsigned distance) : x(x), y(y), distance(distance){}
	PixelDistance() : x(0), y(0), distance(0){}
};

struct PGMCBPC{
	PGMCBPC(vector<PGMImage>& inputImages,
			vector<PGMImage>& outputImages,
			vector<PGMImageError>& errorImages,
			vector<Predictor*>& predictors
			);
	~PGMCBPC();

	bool getStaticPrediction(unsigned i);
	void predict();

private:
	void predict(unsigned imageIndex);
	byte predictElement(unsigned x, unsigned y);

	unsigned distance(unsigned anchorX,unsigned anchorY, unsigned x, unsigned y, unsigned w,unsigned h);
	void insert(PixelDistance pixelDist, PixelDistance similarPixels[M], unsigned* numOfSimilarPixels);

	vector<PGMImage>& inputImages;
	vector<PGMImage>& outputImages;
	vector<PGMImageError>& errorImages;
	vector<Predictor*>& predictors;

	vector<ImageWHSize> imagesMeta;
	vector<byte*> iData;
	vector<byte*> oData;
	vector<byte**> pData;
	vector<short*> eData;

	PixelOffset radiusOffset[R_A];
	PixelOffset vectorOffset[D];
};

#endif /* PGMCBPC_H_ */
