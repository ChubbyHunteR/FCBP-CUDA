#ifndef PREDICTORN_H_
#define PREDICTORN_H_

#include "Predictor.h"

struct PredictorN : public Predictor{
	virtual void cudaPredictAll(void *diData, void *dPredicted, unsigned w, unsigned h);
};

#endif /* PREDICTORN_H_ */
