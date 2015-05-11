#ifndef GRDECODER_H_
#define GRDECODER_H_


#include <fstream>
#include <vector>
#include <memory>
#include <string>
using namespace std;

#include "GRconfig.h"
#include "../PGM/PGMImageError.h"
#include "Bitset.h"

class GRDecoder{

public:
	GRDecoder(ifstream& input, string outputName);
	~GRDecoder();
	shared_ptr<PGMImageError> decode();

private:
	ifstream& input;
	shared_ptr<PGMImageError> output;
	Bitset* data;
	unsigned log2M;

	byte decodeQuotient(unsigned* pos);
	byte decodeRemainder(unsigned* pos);
};


#endif /* GRDECODER_H_ */
