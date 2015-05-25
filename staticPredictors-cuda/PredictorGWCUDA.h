#ifndef PREDICTORGWCUDA_H_
#define PREDICTORGWCUDA_H_

#include "PredictorCUDA.h"

struct PredictorGW : public Predictor{
	virtual void cudaPredictAll(void *diData, void *dPredicted, unsigned w, unsigned h);
};

#endif /* PREDICTORGWCUDA_H_ */
