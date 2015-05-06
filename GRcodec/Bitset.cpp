#include <cstring>

#include "Bitset.h"

#define INIT_SIZE (1024 * 16)

byte masks[8] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};

Bitset::Bitset() : data(new byte[INIT_SIZE]), size(0), capacity_bytes(INIT_SIZE){
}

Bitset::Bitset(byte *data, unsigned size, unsigned capacity) : data(data), size(size), capacity_bytes(capacity / 8){
}

Bitset::~Bitset() {
	delete[] data;
}

void Bitset::push(bool b){
	if(size == capacity_bytes * 8){
		capacity_bytes *= 1.7;
		byte *new_data = new byte[capacity_bytes];
		memcpy(new_data, data, size / 8);
		delete[] data;
		data = new_data;
	}
	++size;
	if(b){
		set(size-1);
	}else{
		reset(size-1);
	}
}

void Bitset::set(unsigned pos){
	if(pos >= size){
		return;
	}
	byte mask = masks[pos % 8];
	data[pos / 8] |= mask;
}

void Bitset::reset(unsigned pos){
	if(pos >= size){
		return;
	}
	byte mask = ~ masks[pos % 8];
	data[pos / 8] &= mask;
}

bool Bitset::operator[](unsigned pos){
	if(pos >= size){
		return false;
	}
	byte mask = masks[pos % 8];
	return data[pos / 8] & mask;
}

byte* Bitset::getData(){
	return data;
}

unsigned Bitset::getSize(){
	return size;
}
