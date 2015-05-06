#include "GRDecoder.h"

GRDecoder::GRDecoder(ifstream& input, string outputName) : input(input), output(), data(nullptr), log2M(0){
	const unsigned b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
	const unsigned S[] = {1, 2, 4, 8, 16};
	unsigned v = M;
	for(int i = 4; i >= 0; --i){
		if(v & b[i]){
			v >>= S[i];
			log2M |= S[i];
		}
	}
	if(M-1 & M){
		++log2M;
	}

	unsigned w, h, maxPixel;
	input >> w >> h >> maxPixel;
	input.ignore();
	output = make_shared<PGMImageError>(outputName.data(), w, h, maxPixel);


	int size = input.tellg();
	input.seekg(0, input.end);
	size = (int)input.tellg() - size;
	input.seekg(-size, input.end);

	byte* data = new byte[size];
	input.read((char*)data, size);
	this->data = new Bitset(data, size * 8, size * 8);
}

GRDecoder::~GRDecoder(){
	delete data;
}

shared_ptr<PGMImageError> GRDecoder::decode(){
	short *out = output->getBuffer();

	//unsigned pos = 2;
	unsigned pos = 0;
	for(unsigned i = 0; i < output->getSize(); ++i){
		byte quotient = decodeQuotient(&pos);
		byte remainder = decodeRemainder(&pos);
		short N = quotient * M + remainder;
		if(N & 0x1){
			++N;
			N = -N;
		}
		N >>= 1;
		out[i] = N;
	}

	return output;
}

byte GRDecoder::decodeQuotient(unsigned* pos){
	byte quotient = 0;

	while((*data)[*pos]){
		++quotient;
		++*pos;
	}
	++*pos;

	return quotient;
}

byte GRDecoder::decodeRemainder(unsigned* pos){
	byte remainder = 0;

	if( !(M-1 & M) ){
		for(int i = 0; i < log2M; ++i){
			remainder <<= 1;
			remainder |= (*data)[*pos];
			++*pos;
		}
	}else{
		unsigned u = log2M - M;
		for(int i = 0; i < log2M - 1; ++i){
			remainder <<= 1;
			remainder &= (*data)[*pos];
			++*pos;
		}
		if(remainder >= u){
			remainder <<= 1;
			remainder &= (*data)[*pos];
			remainder -= u;
			++*pos;
		}
	}

	return remainder;
}
