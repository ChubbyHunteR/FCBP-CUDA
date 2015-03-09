///////////////////////////////////////////////////////////////////////////
// File Name    : PGMImage.cpp
//
// Purpose      : Implementation file for the PGMImage class
//
// Author       : Josip Knezovic (josip.knezovic@fer.hr)
//
//
// Copyright 2005. Josip Knezovic
//
///////////////////////////////////////////////////////////////////////////

#include "PGMImage.h"
//#include "require.h"

#include <cstring>
#include <iostream>		
#include <iomanip>
#include <sstream>
#include <cstdlib>

using namespace std;

///////////////////// construction & destruction ///////////////////////////
////////////////////////////////////////////////////////////////////////////
// Function: PGMImage(const char* imgName)
// 
// Purpose:	Constructor 
//			Takes the image name as the argument and initializes the input file
//			stream imgFile with this name this constructor is used to create 
//          the object that reads the pgm image file (for input)
/////////////////////////////////////////////////////////////////////////////
PGMImage::PGMImage(const char* imgName)
 : modM(ios::in), imgFileM(imgName, ios::in | ios::binary), imgNameM(imgName), heightM(0), pixelMaxM(0), widthM(0)
{	
	// this part simply checks if magic number is included in the file
	// it doesn't check all possible errors in file format
	//char tmp[5];
	if(imgFileM.fail()){
		cerr<<"Opening of input file failed."<<endl;
		exit(EXIT_FAILURE);
	}
	string tmp;
	imgFileM >> tmp;
	string magicNum(tmp);
	if (magicNum != "P5") {
        std::cerr << "Wrong file format " << imgName << endl;
		exit(EXIT_FAILURE);
	}

  string line;
  string comment("#");
  
  //Za Marka:
  //imgFileM >> dummy1 >> dummy2 >> dummy3 >> dummy4>>dummy1 >> dummy2 >> dummy3;

 
 
  //Provjera postoji li linija koja zapocinje s #
  //Ocekuje se komentar u drugoj liniji
  imgFileM>>line;
  while(!isdigit(line[0]))
    imgFileM>>line;
    
    
  istringstream stream(line);
  stream>> widthM; 
  
	// we presume that there's no lines starting with '#' in header
	// read image width, height, and maxPixelValue
	imgFileM >> heightM >> pixelMaxM;
	
	// ignores next char (whitespace)
	imgFileM.ignore();
		
	// buffer pixels into imgBufferM
	imgBufferM = new byte[widthM * heightM];
	imgFileM.read((char*)imgBufferM, widthM * heightM);
}

////////////////////////////////////////////////////////////////////////////////
// Function:	PGMImage(const char*, int width, int height, unsigned pixelMax)
//
// Purpose:		Initializes the PGMImage for output into file. It sets the
//				image width, height, and maximum possible pixel value, allocates
//				the pixel buffer and writes the header into output pgm file.
////////////////////////////////////////////////////////////////////////////////
PGMImage::PGMImage(const char* imgName, unsigned width, unsigned height, byte pixelMax)
 : modM(ios::out), imgFileM(imgName, ios::binary | ios::out), 
   widthM(width), heightM(height), pixelMaxM(pixelMax)
{        
	if(imgFileM.fail()){
		cerr<<"Opening of output file failed."<<endl;
		exit(EXIT_FAILURE);
	}

    imgBufferM = new unsigned char[widthM * heightM];
            
    imgFileM << "P5\n" << widthM << " " << heightM << "\n"
               << pixelMaxM << endl;
    
}

////////////////////////////////////////////////////////////////////////////////
// Function:	~PGMImage()
//
// Purpose:		Destructor
///////////////////////////////////////////////////////////////////////////////
PGMImage::~PGMImage() {
    if (modM == ios::in) {
        // free the allocated buffer
        delete[] imgBufferM;
    } else {
        // if out mode we need to flush the pixel buffer into file 
        imgFileM.write( ((char*)(imgBufferM)), widthM * heightM);    
        // free the allocated buffer
        delete[] imgBufferM;
    } 
}
////////////////// ! construction & destruction ////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// Function: getPixel(int ppos)
// Purpose:  Returns the pixel at the (xPos, yPos) coords. in the pixel array.
//////////////////////////////////////////////////////////////////////////////
byte PGMImage::getPixel(unsigned xPos, unsigned yPos) {
    // check boundaries
    if(xPos >= widthM || yPos >= heightM) {
        cerr << "ERROR: reading PGMImage\n"
             << " xPos = " << xPos << " yPos = " << yPos
             << " width = " << widthM << " height = " << heightM << endl;
        exit(EXIT_FAILURE);
    }
    
    return imgBufferM[yPos * widthM + xPos]; 
}

//////////////////////////////////////////////////////////////////////////////
// Function: getPixel(int pPos)
// Purpose: Returns the pixel at the pPos in the pixel array when the image 
//          is viewed as an array in raster scan order
//////////////////////////////////////////////////////////////////////////////
byte PGMImage::getPixel(unsigned pPos) {
    if((pPos >= widthM * heightM) || (pPos < 0)) {
        cerr << "ERROR: reading PGMImage\n"
             << " pPos = " << pPos << " out of range!" << endl;
        exit(EXIT_FAILURE);
    }
    return imgBufferM[pPos];
}

///////////////////////////////////////////////////////////////////////////////
// Function: readIntoArray(unsigned* buff, int xPos, int yPos, int size)
// Purpose:  Reads the pixels from (xPos, yPos) to (xPos + size, yPos) into
//           the array buff of size size
//////////////////////////////////////////////////////////////////////////////
void PGMImage::readIntoArray(byte* buff, unsigned xPos, unsigned yPos, unsigned size) {
	// no boundary guards for now
	for (int i = 0; i < size; i++) {
		buff[i] = getPixel(xPos++, yPos);
    }
}

///////////////////////////////////////////////////////////////////////////////
// Function: readIntoArray(unsigned* buff, int xPos, int size)
// Purpose:  Reads the pixels from xPos to xPos + size into
//           the array buff of size size
//////////////////////////////////////////////////////////////////////////////
void PGMImage::readIntoArray(byte* buff, unsigned xPos, unsigned size) {
	// no boundary guards for now
	for (int i = 0; i < size; i++) {
		buff[i] = getPixel(xPos++);
    }
}


///////////////////////////////////////////////////////////////////////////////
// Function: writePixel(int, int, unsigned)
// Purpose:  
//////////////////////////////////////////////////////////////////////////////
void PGMImage::writePixel(unsigned xPos, unsigned yPos, byte pixel) {
    if(modM != ios::out) {
        cerr << "ERROR: PGMImage::writePixel()\n"
             << " modM != ios::out " << endl;
        exit(EXIT_FAILURE);
    }
    
    // if coordinates exceed the image width or height - report error
    if ((xPos >= widthM) || (yPos >= heightM)) {
        cerr << "ERROR: reading PGMImage\n"
             << " xPos = " << xPos << " yPos = " << yPos
             << " width = " << widthM << " height = " << heightM << endl;     
        exit(EXIT_FAILURE);
        
    } else {        
        // buffer the pixel in the buffer array
        imgBufferM[yPos * widthM + xPos] = static_cast<unsigned char>(pixel);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Function: imageToText()
// 
// Purpose:  Prints the image into the text file every pixel is printed in readable
// format. It is used for debugging.
//////////////////////////////////////////////////////////////////////////////////
void PGMImage::imageToText() {
	char txtImageName[50];
	std::strcpy(txtImageName, imgNameM.c_str());
	std::strcat(txtImageName, ".txt");

    // create output file stream for text output
	ofstream txtImageFile(txtImageName);
	
	txtImageFile << " image name: " << imgNameM << endl;
	txtImageFile << " width: " << widthM << endl;
	txtImageFile << " height: " << heightM << endl;
	txtImageFile << " PIXELS " << endl;
	txtImageFile << " ---------------------------------------- " << endl;
	
    // print pixels into output text file
	for (int yPos = 0; yPos < getHeight(); yPos++) {
		for (int xPos = 0; xPos < getWidth(); xPos++) {
            txtImageFile << std::setw(5) << getPixel(xPos, yPos);
		}

		txtImageFile << endl;
	}
}
