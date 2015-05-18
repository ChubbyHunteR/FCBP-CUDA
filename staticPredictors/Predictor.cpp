#include "Predictor.h"

void Predictor::predictAll(void *iData, void *predicted, unsigned w, unsigned h){
	for(unsigned y = 0; y < h; ++y){
		for(unsigned x = 0; x < w; ++w){
			predicted[i] = predict(iData, x, y, w, h);
		}
	}
}
