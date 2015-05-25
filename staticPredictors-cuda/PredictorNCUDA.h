#ifndef PREDICTORNCUDA_H_
#define PREDICTORNCUDA_H_

#include "PredictorCUDA.h"

struct PredictorN : public Predictor{
	virtual void cudaPredictAll(void *diData, void *dPredicted, unsigned w, unsigned h);
};

#endif /* PREDICTORNCUDA_H_ */
