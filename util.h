/*
 * util.h
 *
 *  Created on: Apr 15, 2015
 *      Author: lukas
 */

#ifndef UTIL_H_
#define UTIL_H_

struct ImageWHSize{
	unsigned w, h, size;
	ImageWHSize(unsigned w, unsigned h, unsigned size) : w(w), h(h), size(size){}
	ImageWHSize() : w(0), h(0), size(0){}
};

struct PixelOffset{
	int x, y;
	PixelOffset(int x, int y) : x(x), y(y){}
	PixelOffset() : x(0), y(0){}
};


#endif /* UTIL_H_ */
