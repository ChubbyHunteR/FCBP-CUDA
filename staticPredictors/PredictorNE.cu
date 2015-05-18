#include "PredictorNE.h"

byte PredictorNE::predict(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h) {
	++x;
	--y;
	if(x < w && y < h){
		return iData[y * w + x];
	}
	return 0;
}
