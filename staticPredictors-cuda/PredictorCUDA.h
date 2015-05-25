#ifndef PREDICTORCUDA_H_
#define PREDICTORCUDA_H_
typedef unsigned char byte;

struct Predictor{
	virtual void cudaPredictAll(void *diData, void *dPredicted, unsigned w, unsigned h) = 0;
};

#endif /* PREDICTORCUDA_H_ */
