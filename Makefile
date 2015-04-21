CC = nvcc
CFLAGS = -O3 -std=c++11
CFLAGS_DEBUG = -g -G -std=c++11 -DDEBUG


##################
# COMMON OBJECTS #
##################

COMMON_OBJECTS = PredictorN.o PredictorNW.o PredictorGW.o PredictorW.o PredictorNE.o PredictorGN.o PredictorPL.o PGMImage.o PGMImageError.o
COMMON_OBJECTS_DEBUG = PredictorN_d.o PredictorNW_d.o PredictorGW_d.o PredictorW_d.o PredictorNE_d.o PredictorGN_d.o PredictorPL_d.o PGMImage_d.o PGMImageError_d.o 


#################
# CODER OBJECTS #
#################
CODER = cbpc
CODER_DEBUG = cbpc_d
CODER_OBJECTS = PGMCBPCCUDA.o cbpc.o
CODER_OBJECTS_DEBUG = PGMCBPCCUDA_d.o cbpc_d.o


###################
# DECODER OBJECTS #
###################
DECODER = cbpd
DECODER_DEBUG = cbpd_d
DECODER_OBJECTS = PGMCBPDCUDA.o cbpd.o
DECODER_OBJECTS_DEBUG = PGMCBPDCUDA_d.o cbpd_d.o


##################
# COMMON RELEASE #
##################

all: $(CODER) $(DECODER)

PredictorN.o: predictors/PredictorN.cu predictors/PredictorN.h
	$(CC) -c -o PredictorN.o $(CFLAGS) predictors/PredictorN.cu

PredictorNW.o: predictors/PredictorNW.cu predictors/PredictorNW.h
	$(CC) -c -o PredictorNW.o $(CFLAGS) predictors/PredictorNW.cu

PredictorGW.o: predictors/PredictorGW.cu predictors/PredictorGW.h
	$(CC) -c -o PredictorGW.o $(CFLAGS) predictors/PredictorGW.cu

PredictorW.o: predictors/PredictorW.cu predictors/PredictorW.h
	$(CC) -c -o PredictorW.o $(CFLAGS) predictors/PredictorW.cu

PredictorNE.o: predictors/PredictorNE.cu predictors/PredictorNE.h
	$(CC) -c -o PredictorNE.o $(CFLAGS) predictors/PredictorNE.cu

PredictorGN.o: predictors/PredictorGN.cu predictors/PredictorGN.h
	$(CC) -c -o PredictorGN.o $(CFLAGS) predictors/PredictorGN.cu

PredictorPL.o: predictors/PredictorPL.cu predictors/PredictorPL.h
	$(CC) -c -o PredictorPL.o $(CFLAGS) predictors/PredictorPL.cu

PGMImage.o: PGMImage.cpp PGMImage.h
	$(CC) -c -o PGMImage.o $(CFLAGS) PGMImage.cpp

PGMImageError.o: PGMImageError.cpp PGMImageError.h
	$(CC) -c -o PGMImageError.o $(CFLAGS) PGMImageError.cpp


################
# COMMON DEBUG #
################

debug: $(CODER_DEBUG) $(DECODER_DEBUG)

PredictorN_d.o: predictors/PredictorN.cu predictors/PredictorN.h
	$(CC) -c -o PredictorN_d.o $(CFLAGS_DEBUG) predictors/PredictorN.cu

PredictorNW_d.o: predictors/PredictorNW.cu predictors/PredictorNW.h
	$(CC) -c -o PredictorNW_d.o $(CFLAGS_DEBUG) predictors/PredictorNW.cu

PredictorGW_d.o: predictors/PredictorGW.cu predictors/PredictorGW.h
	$(CC) -c -o PredictorGW_d.o $(CFLAGS_DEBUG) predictors/PredictorGW.cu

PredictorW_d.o: predictors/PredictorW.cu predictors/PredictorW.h
	$(CC) -c -o PredictorW_d.o $(CFLAGS_DEBUG) predictors/PredictorW.cu

PredictorNE_d.o: predictors/PredictorNE.cu predictors/PredictorNE.h
	$(CC) -c -o PredictorNE_d.o $(CFLAGS_DEBUG) predictors/PredictorNE.cu

PredictorGN_d.o: predictors/PredictorGN.cu predictors/PredictorGN.h
	$(CC) -c -o PredictorGN_d.o $(CFLAGS_DEBUG) predictors/PredictorGN.cu

PredictorPL_d.o: predictors/PredictorPL.cu predictors/PredictorPL.h
	$(CC) -c -o PredictorPL_d.o $(CFLAGS_DEBUG) predictors/PredictorPL.cu

PGMImage_d.o: PGMImage.cpp PGMImage.h
	$(CC) -c -o PGMImage_d.o $(CFLAGS_DEBUG) PGMImage.cpp

PGMImageError_d.o: PGMImageError.cpp PGMImageError.h
	$(CC) -c -o PGMImageError_d.o $(CFLAGS_DEBUG) PGMImageError.cpp


#################
# CODER RELEASE #
#################

$(CODER): $(COMMON_OBJECTS) $(CODER_OBJECTS)
	$(CC) -o $(CODER) $(CFLAGS) $(COMMON_OBJECTS) $(CODER_OBJECTS)

PGMCBPCCUDA.o: PGMCBPCCUDA.cu PGMCBPCCUDA.h config.h util.h
	$(CC) -c -o PGMCBPCCUDA.o $(CFLAGS) PGMCBPCCUDA.cu

cbpc.o: cbpc.cpp config.h
	$(CC) -c -o cbpc.o $(CFLAGS) cbpc.cpp


###############
# CODER DEBUG #
###############

$(CODER_DEBUG): $(COMMON_OBJECTS_DEBUG) $(CODER_OBJECTS_DEBUG)
	$(CC) -o $(CODER_DEBUG) $(CFLAGS_DEBUG) $(COMMON_OBJECTS_DEBUG) $(CODER_OBJECTS_DEBUG)

PGMCBPCCUDA_d.o: PGMCBPCCUDA.cu PGMCBPCCUDA.h config.h util.h
	$(CC) -c -o PGMCBPCCUDA_d.o $(CFLAGS_DEBUG) PGMCBPCCUDA.cu

cbpc_d.o: cbpc.cpp config.h
	$(CC) -c -o cbpc_d.o $(CFLAGS_DEBUG) cbpc.cpp


###################
# DECODER RELEASE #
###################

$(DECODER): $(COMMON_OBJECTS) $(DECODER_OBJECTS)
	$(CC) -o $(DECODER) $(CFLAGS) $(COMMON_OBJECTS) $(DECODER_OBJECTS)

PGMCBPDCUDA.o: PGMCBPDCUDA.cu PGMCBPDCUDA.h config.h util.h
	$(CC) -c -o PGMCBPDCUDA.o $(CFLAGS) PGMCBPDCUDA.cu

cbpd.o: cbpd.cpp config.h
	$(CC) -c -o cbpd.o $(CFLAGS) cbpd.cpp


#################
# DECODER DEBUG #
#################

$(DECODER_DEBUG): $(COMMON_OBJECTS_DEBUG) $(DECODER_OBJECTS_DEBUG)
	$(CC) -o $(DECODER_DEBUG) $(CFLAGS_DEBUG) $(COMMON_OBJECTS_DEBUG) $(DECODER_OBJECTS_DEBUG)

PGMCBPDCUDA_d.o: PGMCBPDCUDA.cu PGMCBPDCUDA.h config.h util.h
	$(CC) -c -o PGMCBPDCUDA_d.o $(CFLAGS_DEBUG) PGMCBPDCUDA.cu

cbpd_d.o: cbpd.cpp config.h
	$(CC) -c -o cbpd_d.o $(CFLAGS_DEBUG) cbpd.cpp



clean:
	-rm -f $(CODER) $(CODER_DEBUG) $(DECODER) $(DECODER_DEBUG) *.o
