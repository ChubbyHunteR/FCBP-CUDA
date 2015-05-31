FCBP CUDA
=========

CBPC implementation in CUDA. FER undergraduate thesis, 2014/2015

This project depends on CUDA Toolkit (or the correct drivers, a CUDA capable graphics card and nvcc).

Release build:
make

Debug build:
make debug

Clean:
make clean

Directory structure
===================
cbpc - CBP coder
cbpc-cuda - CBP coder, CUDA
cbpd - CBP decoder
cbpd-cuda - CBP decoder, CUDA
cbpcTester - CBPC tester
GRcodec - Golomb Rice coder and decoder
PGM - Classes that handle PGM images and the 'error' file, which is a file that holds the difference between a prediction and the original.
staticPredictors - Static predictors that are weighted before they produce a final prediction
staticPredictors-cuda - Static predictors that are weighted before they produce a final prediction, CUDA
