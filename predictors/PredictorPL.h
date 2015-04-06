#ifndef PREDICTORPL_H_
#define PREDICTORPL_H_

#include "Predictor.h"

struct PredictorPL : public Predictor{
	virtual void predict(void *diData, void *dPredicted, unsigned w, unsigned h);
};

#endif /* PREDICTORPL_H_ */
