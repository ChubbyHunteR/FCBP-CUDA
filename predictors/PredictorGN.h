#ifndef PREDICTORGN_H_
#define PREDICTORGN_H_

#include "Predictor.h"

struct PredictorGN : public Predictor{
	virtual void cudaPredictAll(void *diData, void *dPredicted, unsigned w, unsigned h);
	virtual byte predict(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h);
};

#endif /* PREDICTORGN_H_ */
