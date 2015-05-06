#ifndef BITSET_H_
#define BITSET_H_

typedef unsigned char byte;

class Bitset {
public:
	Bitset();
	Bitset(byte *data, unsigned size, unsigned capacity);
	~Bitset();

	void push(bool b);
	void set(unsigned pos);
	void reset(unsigned pos);
	bool operator[](unsigned pos);
	byte* getData();
	unsigned getSize();

private:
	byte *data;
	unsigned size;
	unsigned capacity_bytes;
};

#endif /* BITSET_H_ */
