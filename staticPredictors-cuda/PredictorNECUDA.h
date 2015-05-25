#ifndef PREDICTORNECUDA_H_
#define PREDICTORNECUDA_H_

#include "PredictorCUDA.h"

struct PredictorNE : public Predictor{
	virtual void cudaPredictAll(void *diData, void *dPredicted, unsigned w, unsigned h);
};

#endif /* PREDICTORNECUDA_H_ */
