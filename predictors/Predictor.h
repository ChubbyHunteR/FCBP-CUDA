#ifndef PREDICTOR_H_
#define PREDICTOR_H_
typedef unsigned char byte;

struct Predictor{
	virtual void cudaPredictAll(void *diData, void *dPredicted, unsigned w, unsigned h) = 0;
	virtual byte predict(byte *iData, unsigned anchorX, unsigned anchorY, unsigned w, unsigned h) = 0;
};

#endif /* PREDICTOR_H_ */
