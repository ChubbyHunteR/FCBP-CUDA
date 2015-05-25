#include "PredictorN.h"

byte PredictorN::predict(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h) {
	--y;
	if(x < w && y < h){
		return iData[y * w + x];
	}
	return 0;
}
