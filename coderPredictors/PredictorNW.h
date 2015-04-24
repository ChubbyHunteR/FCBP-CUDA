#ifndef PREDICTORNW_H_
#define PREDICTORNW_H_

#include "Predictor.h"

struct PredictorNW : public Predictor{
	virtual void cudaPredictAll(void *diData, void *dPredicted, unsigned w, unsigned h);
};

#endif /* PREDICTORNW_H_ */
