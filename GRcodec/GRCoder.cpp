#include "GRCoder.h"

GRCoder::GRCoder(PGMImageError& input, ofstream& output) : input(input), output(output), data(), log2M(0){
//	data.push(0);
//	data.push(0);
//	data.push(0);

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
}

void GRCoder::encode(){
	short *in = input.getBuffer();

	for(unsigned i = 0; i < input.getSize(); ++i){
		short N = in[i];
		if(N >= 0){
			N *= 2;
		}else{
			N *= -2;
			--N;
		}
		if(N != 0){
			N <<= 1;
			N >>= 1;
		}
		unsigned quotient = N / M;
		unsigned remainder = N % M;

		codeQuotient(quotient);
		codeRemainder(remainder);
	}

	byte numOfEmptyBits = data.getSize() % 8;
	for(; numOfEmptyBits > 0; --numOfEmptyBits){
		data.push(0);
	}
//	if(numOfEmptyBits & 0x1){
//		data.set(2);
//	}
//	if(numOfEmptyBits & 0x2){
//		data.set(1);
//	}
//	if(numOfEmptyBits & 0x4){
//		data.set(0);
//	}

	writeData();
}

void GRCoder::codeQuotient(unsigned quotient){
	for(; quotient > 0; --quotient){
		data.push(1);
	}
	data.push(0);
}

void GRCoder::codeRemainder(unsigned remainder){
	unsigned mask = 1 << log2M - 1;

	if( !(M-1 & M) ){
		while(mask){
			data.push(remainder & mask);
			mask >>= 1;
		}
	}else if(remainder < (1 << log2M) - M){
		mask >>= 1;
		while(mask){
			data.push(remainder & mask);
			mask >>= 1;
		}
	}else{
		remainder += (1 << log2M) - M;
		while(mask){
			data.push(remainder & mask);
			mask >>= 1;
		}
	}
}

void GRCoder::writeData(){
	output << input.getWidth() << " " << input.getHeight() << " " << input.getPixelMax() << " ";
	unsigned size = data.getSize() / 8;
	if(data.getSize() % 8 != 0){
		++size;
	}
	output.write((const char*)data.getData(), size);
}
