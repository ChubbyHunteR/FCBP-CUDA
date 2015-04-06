#ifndef PREDICTORNE_H_
#define PREDICTORNE_H_

#include "Predictor.h"

struct PredictorNE : public Predictor{
	virtual void predict(void *diData, void *dPredicted, unsigned w, unsigned h);
};

#endif /* PREDICTORNE_H_ */
