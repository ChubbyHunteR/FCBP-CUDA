#ifndef PREDICTOR_H_
#define PREDICTOR_H_

struct Predictor{
	virtual void predict(void *diData, void *dPredicted, unsigned w, unsigned h) = 0;
};

#endif /* PREDICTOR_H_ */
