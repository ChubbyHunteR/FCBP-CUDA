#ifndef PREDICTOR_H_
#define PREDICTOR_H_

struct Predictor{
	virtual void predict(void *diData, void *dPredicted, void* dLookupOffsetX, void* dLookupOffsetY, unsigned w, unsigned h);
};

#endif /* PREDICTOR_H_ */
