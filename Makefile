CC = nvcc
CFLAGS = -O3 -std=c++11 -code sm_50 -arch compute_50
CFLAGS_DEBUG = -g -G -std=c++11 -code sm_50 -arch compute_50 -DDEBUG


##################
# COMMON OBJECTS #
##################

COMMON_OBJECTS = PGMImage.o PGMImageError.o
COMMON_OBJECTS_DEBUG = PGMImage_d.o PGMImageError_d.o

#####################
# GOLOMB RICE CODEC #
#####################

GRC = grc
GRC_DEBUG = grc_d
GRD = grd
GRD_DEBUG = grd_d 


#################
# CODER OBJECTS #
#################
CODER = cbpc
CODER_DEBUG = cbpc_d
CODER_OBJECTS = PredictorN.o PredictorNW.o PredictorGW.o PredictorW.o PredictorNE.o PredictorGN.o PredictorPL.o PGMCBPCCUDA.o cbpc.o
CODER_OBJECTS_DEBUG = PredictorN_d.o PredictorNW_d.o PredictorGW_d.o PredictorW_d.o PredictorNE_d.o PredictorGN_d.o PredictorPL_d.o PGMCBPCCUDA_d.o cbpc_d.o


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

all: $(CODER) $(DECODER) $(GRC) $(GRD)

PGMImage.o: PGM/PGMImage.cpp PGM/PGMImage.h
	$(CC) -c -o PGMImage.o $(CFLAGS) PGM/PGMImage.cpp

PGMImageError.o: PGM/PGMImageError.cpp PGM/PGMImageError.h
	$(CC) -c -o PGMImageError.o $(CFLAGS) PGM/PGMImageError.cpp


################
# COMMON DEBUG #
################

debug: $(CODER_DEBUG) $(DECODER_DEBUG) $(GRC_DEBUG) $(GRD_DEBUG)

PGMImage_d.o: PGM/PGMImage.cpp PGM/PGMImage.h
	$(CC) -c -o PGMImage_d.o $(CFLAGS_DEBUG) PGM/PGMImage.cpp

PGMImageError_d.o: PGM/PGMImageError.cpp PGM/PGMImageError.h
	$(CC) -c -o PGMImageError_d.o $(CFLAGS_DEBUG) PGM/PGMImageError.cpp


#################
# CODER RELEASE #
#################

$(CODER): $(COMMON_OBJECTS) $(CODER_OBJECTS)
	$(CC) -o $(CODER) $(CFLAGS) $(COMMON_OBJECTS) $(CODER_OBJECTS)

PredictorN.o: staticPredictors-cuda/PredictorN.cu staticPredictors-cuda/PredictorN.h
	$(CC) -c -o PredictorN.o $(CFLAGS) staticPredictors-cuda/PredictorN.cu

PredictorNW.o: staticPredictors-cuda/PredictorNW.cu staticPredictors-cuda/PredictorNW.h
	$(CC) -c -o PredictorNW.o $(CFLAGS) staticPredictors-cuda/PredictorNW.cu

PredictorGW.o: staticPredictors-cuda/PredictorGW.cu staticPredictors-cuda/PredictorGW.h
	$(CC) -c -o PredictorGW.o $(CFLAGS) staticPredictors-cuda/PredictorGW.cu

PredictorW.o: staticPredictors-cuda/PredictorW.cu staticPredictors-cuda/PredictorW.h
	$(CC) -c -o PredictorW.o $(CFLAGS) staticPredictors-cuda/PredictorW.cu

PredictorNE.o: staticPredictors-cuda/PredictorNE.cu staticPredictors-cuda/PredictorNE.h
	$(CC) -c -o PredictorNE.o $(CFLAGS) staticPredictors-cuda/PredictorNE.cu

PredictorGN.o: staticPredictors-cuda/PredictorGN.cu staticPredictors-cuda/PredictorGN.h
	$(CC) -c -o PredictorGN.o $(CFLAGS) staticPredictors-cuda/PredictorGN.cu

PredictorPL.o: staticPredictors-cuda/PredictorPL.cu staticPredictors-cuda/PredictorPL.h
	$(CC) -c -o PredictorPL.o $(CFLAGS) staticPredictors-cuda/PredictorPL.cu

PGMCBPCCUDA.o: cbpc-cuda/PGMCBPCCUDA.cu cbpc-cuda/PGMCBPCCUDA.h config.h util.h
	$(CC) -c -o PGMCBPCCUDA.o $(CFLAGS) cbpc-cuda/PGMCBPCCUDA.cu

cbpc.o: cbpc-cuda/cbpc.cpp config.h
	$(CC) -c -o cbpc.o $(CFLAGS) cbpc-cuda/cbpc.cpp


###############
# CODER DEBUG #
###############

$(CODER_DEBUG): $(COMMON_OBJECTS_DEBUG) $(CODER_OBJECTS_DEBUG)
	$(CC) -o $(CODER_DEBUG) $(CFLAGS_DEBUG) $(COMMON_OBJECTS_DEBUG) $(CODER_OBJECTS_DEBUG)

PredictorN_d.o: staticPredictors-cuda/PredictorN.cu staticPredictors-cuda/PredictorN.h
	$(CC) -c -o PredictorN_d.o $(CFLAGS_DEBUG) staticPredictors-cuda/PredictorN.cu

PredictorNW_d.o: staticPredictors-cuda/PredictorNW.cu staticPredictors-cuda/PredictorNW.h
	$(CC) -c -o PredictorNW_d.o $(CFLAGS_DEBUG) staticPredictors-cuda/PredictorNW.cu

PredictorGW_d.o: staticPredictors-cuda/PredictorGW.cu staticPredictors-cuda/PredictorGW.h
	$(CC) -c -o PredictorGW_d.o $(CFLAGS_DEBUG) staticPredictors-cuda/PredictorGW.cu

PredictorW_d.o: staticPredictors-cuda/PredictorW.cu staticPredictors-cuda/PredictorW.h
	$(CC) -c -o PredictorW_d.o $(CFLAGS_DEBUG) staticPredictors-cuda/PredictorW.cu

PredictorNE_d.o: staticPredictors-cuda/PredictorNE.cu staticPredictors-cuda/PredictorNE.h
	$(CC) -c -o PredictorNE_d.o $(CFLAGS_DEBUG) staticPredictors-cuda/PredictorNE.cu

PredictorGN_d.o: staticPredictors-cuda/PredictorGN.cu staticPredictors-cuda/PredictorGN.h
	$(CC) -c -o PredictorGN_d.o $(CFLAGS_DEBUG) staticPredictors-cuda/PredictorGN.cu

PredictorPL_d.o: staticPredictors-cuda/PredictorPL.cu staticPredictors-cuda/PredictorPL.h
	$(CC) -c -o PredictorPL_d.o $(CFLAGS_DEBUG) staticPredictors-cuda/PredictorPL.cu

PGMCBPCCUDA_d.o: cbpc-cuda/PGMCBPCCUDA.cu cbpc-cuda/PGMCBPCCUDA.h config.h util.h
	$(CC) -c -o PGMCBPCCUDA_d.o $(CFLAGS_DEBUG) cbpc-cuda/PGMCBPCCUDA.cu

cbpc_d.o: cbpc-cuda/cbpc.cpp config.h
	$(CC) -c -o cbpc_d.o $(CFLAGS_DEBUG) cbpc-cuda/cbpc.cpp


###################
# DECODER RELEASE #
###################

$(DECODER): $(COMMON_OBJECTS) $(DECODER_OBJECTS)
	$(CC) -o $(DECODER) $(CFLAGS) $(COMMON_OBJECTS) $(DECODER_OBJECTS)

PGMCBPDCUDA.o: cbpd-cuda/PGMCBPDCUDA.cu cbpd-cuda/PGMCBPDCUDA.h config.h util.h
	$(CC) -c -o PGMCBPDCUDA.o $(CFLAGS) cbpd-cuda/PGMCBPDCUDA.cu

cbpd.o: cbpd-cuda/cbpd.cpp config.h
	$(CC) -c -o cbpd.o $(CFLAGS) cbpd-cuda/cbpd.cpp


#################
# DECODER DEBUG #
#################

$(DECODER_DEBUG): $(COMMON_OBJECTS_DEBUG) $(DECODER_OBJECTS_DEBUG)
	$(CC) -o $(DECODER_DEBUG) $(CFLAGS_DEBUG) $(COMMON_OBJECTS_DEBUG) $(DECODER_OBJECTS_DEBUG)

PGMCBPDCUDA_d.o: cbpd-cuda/PGMCBPDCUDA.cu cbpd-cuda/PGMCBPDCUDA.h config.h util.h
	$(CC) -c -o PGMCBPDCUDA_d.o $(CFLAGS_DEBUG) cbpd-cuda/PGMCBPDCUDA.cu

cbpd_d.o: cbpd-cuda/cbpd.cpp config.h
	$(CC) -c -o cbpd_d.o $(CFLAGS_DEBUG) cbpd-cuda/cbpd.cpp


################
# CODER TESTER #
################

tester: cbpcTester/cbpcTester.cpp $(COMMON_OBJECTS)
	$(CC) -o ct $(CFLAGS) cbpcTester/cbpcTester.cpp $(COMMON_OBJECTS)

tester_d: cbpcTester/cbpcTester.cpp $(COMMON_OBJECTS_DEBUG)
	$(CC) -o ct_d $(CFLAGS_DEBUG) cbpcTester/cbpcTester.cpp $(COMMON_OBJECTS_DEBUG)


############
# GR CODER #
############

grc: GRcodec/grc.cpp GRCoder.o PGMImageError.o Bitset.o
	$(CC) -o grc $(CFLAGS) GRcodec/grc.cpp GRCoder.o PGMImageError.o Bitset.o

GRCoder.o: GRcodec/GRCoder.cpp GRcodec/GRCoder.h GRcodec/GRconfig.h
	$(CC) -c -o GRCoder.o $(CFLAGS) GRcodec/GRCoder.cpp

Bitset.o: GRcodec/Bitset.cpp GRcodec/Bitset.h
	$(CC) -c -o Bitset.o $(CFLAGS) GRcodec/Bitset.cpp


##################
# GR CODER DEBUG #
##################

grc_d: GRcodec/grc.cpp PGMImageError_d.o GRCoder_d.o Bitset_d.o
	$(CC) -o grc_d $(CFLAGS_DEBUG) GRcodec/grc.cpp PGMImageError_d.o GRCoder_d.o Bitset_d.o

GRCoder_d.o: GRcodec/GRCoder.cpp GRcodec/GRCoder.h GRcodec/GRconfig.h
	$(CC) -c -o GRCoder_d.o $(CFLAGS_DEBUG) GRcodec/GRCoder.cpp

Bitset_d.o: GRcodec/Bitset.cpp GRcodec/Bitset.h
	$(CC) -c -o Bitset_d.o $(CFLAGS_DEBUG) GRcodec/Bitset.cpp


##############
# GR DECODER #
##############

grd: GRcodec/grd.cpp GRDecoder.o PGMImageError.o Bitset.o
	$(CC) -o grd $(CFLAGS) GRcodec/grd.cpp GRDecoder.o PGMImageError.o Bitset.o

GRDecoder.o: GRcodec/GRDecoder.cpp GRcodec/GRDecoder.h GRcodec/GRconfig.h
	$(CC) -c -o GRDecoder.o $(CFLAGS) GRcodec/GRDecoder.cpp


####################
# GR DECODER DEBUG #
####################

grd_d: GRcodec/grd.cpp PGMImageError_d.o GRDecoder_d.o Bitset_d.o
	$(CC) -o grd_d $(CFLAGS_DEBUG) GRcodec/grd.cpp PGMImageError_d.o GRDecoder_d.o Bitset_d.o

GRDecoder_d.o: GRcodec/GRDecoder.cpp GRcodec/GRDecoder.h GRcodec/GRconfig.h
	$(CC) -c -o GRDecoder_d.o $(CFLAGS_DEBUG) GRcodec/GRDecoder.cpp



clean:
	-rm -f $(CODER) $(CODER_DEBUG) $(DECODER) $(DECODER_DEBUG) $(GRC) $(GRC_DEBUG) $(GRD) $(GRD_DEBUG) *.o ct ct_d
