#include "PredictorW.h"

byte PredictorW::predict(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h) {
	--x;
	if(x < w && y < h){
		return iData[y * w + x];
	}
	return 0;
}
