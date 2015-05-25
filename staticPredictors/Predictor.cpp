#include "Predictor.h"

void Predictor::predictAll(byte *iData, byte *predicted, unsigned w, unsigned h){
	for(unsigned y = 0; y < h; ++y){
		for(unsigned x = 0; x < w; ++w){
			predicted[x + y * w] = predict(iData, x, y, w, h);
		}
	}
}
