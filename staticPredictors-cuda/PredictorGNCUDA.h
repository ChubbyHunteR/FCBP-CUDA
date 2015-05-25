#ifndef PREDICTORGNCUDA_H_
#define PREDICTORGNCUDA_H_

#include "PredictorCUDA.h"

struct PredictorGN : public Predictor{
	virtual void cudaPredictAll(void *diData, void *dPredicted, unsigned w, unsigned h);
};

#endif /* PREDICTORGNCUDA_H_ */
