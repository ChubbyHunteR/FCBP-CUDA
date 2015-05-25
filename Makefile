CC = nvcc
CFLAGS = -O3 -std=c++11 -code sm_50 -arch compute_50
CFLAGS_DEBUG = -g -G -std=c++11 -code sm_50 -arch compute_50 -DDEBUG

#############
# VARIABLES #
#############

CODER_TESTER = ct
CODER_TESTER_DEBUG = ct_d
CODER_TESTER_OBJECTS = PGMImage.o PGMImageError.o
CODER_TESTER_OBJECTS_DEBUG = PGMImage_d.o PGMImageError_d.o

GRC = grc
GRC_DEBUG = grc_d
GRC_OBJECTS = GRCoder.o PGMImageError.o Bitset.o
GRC_OBJECTS_DEBUG = PGMImageError_d.o GRCoder_d.o Bitset_d.o

GRD = grd
GRD_DEBUG = grd_d
GRD_OBJECTS = GRDecoder.o PGMImageError.o Bitset.o
GRD_OBJECTS_DEBUG = PGMImageError_d.o GRDecoder_d.o Bitset_d.o

CODER_CUDA = cbpcoder-cuda
CODER_CUDA_DEBUG = cbpcoder-cuda_d
CODER_CUDA_OBJECTS =	PredictorNCUDA.o PredictorNWCUDA.o PredictorGWCUDA.o PredictorWCUDA.o PredictorNECUDA.o PredictorGNCUDA.o PredictorPLCUDA.o\
						PGMCBPCCUDA.o PGMImage.o PGMImageError.o
CODER_CUDA_OBJECTS_DEBUG =	PredictorNCUDA_d.o PredictorNWCUDA_d.o PredictorGWCUDA_d.o PredictorWCUDA_d.o PredictorNECUDA_d.o PredictorGNCUDA_d.o PredictorPLCUDA_d.o\
							PGMCBPCCUDA_d.o PGMImage_d.o PGMImageError_d.o

CODER = cbpcoder
CODER_DEBUG = cbpcoder_d
CODER_OBJECTS =	PredictorN.o PredictorNW.o PredictorGW.o PredictorW.o PredictorNE.o PredictorGN.o PredictorPL.o\
				Predictor.o PGMCBPC.o PGMImage.o PGMImageError.o
CODER_OBJECTS_DEBUG =	PredictorN_d.o PredictorNW_d.o PredictorGW_d.o PredictorW_d.o PredictorNE_d.o PredictorGN_d.o PredictorPL_d.o\
						Predictor.o PGMCBPC_d.o PGMImage_d.o PGMImageError_d.o

DECODER_CUDA = cbpdecoder-cuda
DECODER_CUDA_DEBUG = cbpdecoder-cuda_d
DECODER_CUDA_OBJECTS = PGMCBPDCUDA.o PGMImage.o PGMImageError.o
DECODER_CUDA_OBJECTS_DEBUG = PGMCBPDCUDA_d.o PGMImage_d.o PGMImageError_d.o

DECODER = cbpdecoder
DECODER_DEBUG = cbpdecoder_d
DECODER_OBJECTS =	PredictorN.o PredictorNW.o PredictorGW.o PredictorW.o PredictorNE.o PredictorGN.o PredictorPL.o\
					Predictor.o PGMCBPD.o PGMImage.o PGMImageError.o
DECODER_OBJECTS_DEBUG =	PredictorN_d.o PredictorNW_d.o PredictorGW_d.o PredictorW_d.o PredictorNE_d.o PredictorGN_d.o PredictorPL_d.o\
						Predictor_d.o PGMCBPD_d.o PGMImage.o PGMImageError.o

all: $(CODER) $(DECODER) $(CODER_CUDA) $(DECODER_CUDA) $(GRC) $(GRD) $(CODER_TESTER)

debug: $(CODER_DEBUG) $(DECODER_DEBUG) $(CODER_CUDA_DEBUG) $(DECODER_CUDA_DEBUG) $(GRC_DEBUG) $(GRD_DEBUG) $(CODER_TESTER_DEBUG)

clean:
	-rm -f 	$(CODER) $(CODER_DEBUG) $(DECODER) $(DECODER_DEBUG)\
			$(CODER_CUDA) $(CODER_CUDA_DEBUG) $(DECODER_CUDA) $(DECODER_CUDA_DEBUG)\
			$(GRC) $(GRC_DEBUG) $(GRD) $(GRD_DEBUG)\
			$(CODER_TESTER) $(CODER_TESTER_DEBUG)\
			*.o


####################
# PGMIMAGE OBJECTS #
####################

PGMImage.o: PGM/PGMImage.cpp PGM/PGMImage.h
	$(CC) -c -o $@ $(CFLAGS) $<

PGMImageError.o: PGM/PGMImageError.cpp PGM/PGMImageError.h
	$(CC) -c -o $@ $(CFLAGS) $<

PGMImage_d.o: PGM/PGMImage.cpp PGM/PGMImage.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<

PGMImageError_d.o: PGM/PGMImageError.cpp PGM/PGMImageError.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<


################
# CODER TESTER #
################

ct: cbpcTester/cbpcTester.cpp $(CODER_TESTER_OBJECTS)
	$(CC) -o $@ $(CFLAGS) $< $(CODER_TESTER_OBJECTS)

ct_d: cbpcTester/cbpcTester.cpp $(CODER_TESTER_OBJECTS_DEBUG)
	$(CC) -o $@ $(CFLAGS_DEBUG) $< $(CODER_TESTER_OBJECTS_DEBUG)


#####################
# GOLOMB RICE CODEC #
#####################

$(GRC): GRcodec/grc.cpp $(GRC_OBJECTS)
	$(CC) -o $@ $(CFLAGS) $< $(GRC_OBJECTS)

GRCoder.o: GRcodec/GRCoder.cpp GRcodec/GRCoder.h GRcodec/GRconfig.h
	$(CC) -c -o $@ $(CFLAGS) $<

Bitset.o: GRcodec/Bitset.cpp GRcodec/Bitset.h
	$(CC) -c -o $@ $(CFLAGS) $<

$(GRC_DEBUG): GRcodec/grc.cpp $(GRC_OBJECTS_DEBUG)
	$(CC) -o $@ $(CFLAGS_DEBUG) $< $(GRC_OBJECTS_DEBUG)

GRCoder_d.o: GRcodec/GRCoder.cpp GRcodec/GRCoder.h GRcodec/GRconfig.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<

Bitset_d.o: GRcodec/Bitset.cpp GRcodec/Bitset.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<

$(GRD): GRcodec/grd.cpp $(GRD_OBJECTS)
	$(CC) -o $@ $(CFLAGS) $< $(GRD_OBJECTS)

GRDecoder.o: GRcodec/GRDecoder.cpp GRcodec/GRDecoder.h GRcodec/GRconfig.h
	$(CC) -c -o $@ $(CFLAGS) $<

$(GRD_DEBUG): GRcodec/grd.cpp $(GRD_OBJECTS_DEBUG)
	$(CC) -o $@ $(CFLAGS_DEBUG) $< $(GRD_OBJECTS_DEBUG)

GRDecoder_d.o: GRcodec/GRDecoder.cpp GRcodec/GRDecoder.h GRcodec/GRconfig.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<


#########
# CODER #
#########

$(CODER_CUDA): cbpc-cuda/cbpc.cpp $(CODER_CUDA_OBJECTS) config.h
	$(CC) -o $@ $(CFLAGS) $< $(CODER_CUDA_OBJECTS)

PredictorNCUDA.o: staticPredictors-cuda/PredictorNCUDA.cu staticPredictors-cuda/PredictorNCUDA.h
	$(CC) -c -o $@ $(CFLAGS) $<

PredictorNWCUDA.o: staticPredictors-cuda/PredictorNWCUDA.cu staticPredictors-cuda/PredictorNWCUDA.h
	$(CC) -c -o $@ $(CFLAGS) $<

PredictorGWCUDA.o: staticPredictors-cuda/PredictorGWCUDA.cu staticPredictors-cuda/PredictorGWCUDA.h
	$(CC) -c -o $@ $(CFLAGS) $<

PredictorWCUDA.o: staticPredictors-cuda/PredictorWCUDA.cu staticPredictors-cuda/PredictorWCUDA.h
	$(CC) -c -o $@ $(CFLAGS) $<

PredictorNECUDA.o: staticPredictors-cuda/PredictorNECUDA.cu staticPredictors-cuda/PredictorNECUDA.h
	$(CC) -c -o $@ $(CFLAGS) $<

PredictorGNCUDA.o: staticPredictors-cuda/PredictorGNCUDA.cu staticPredictors-cuda/PredictorGNCUDA.h
	$(CC) -c -o $@ $(CFLAGS) $<

PredictorPLCUDA.o: staticPredictors-cuda/PredictorPLCUDA.cu staticPredictors-cuda/PredictorPLCUDA.h
	$(CC) -c -o $@ $(CFLAGS) $<

PGMCBPCCUDA.o: cbpc-cuda/PGMCBPCCUDA.cu cbpc-cuda/PGMCBPCCUDA.h config.h util.h
	$(CC) -c -o $@ $(CFLAGS) $<

$(CODER_CUDA_DEBUG): cbpc-cuda/cbpc.cpp $(CODER_CUDA_OBJECTS_DEBUG)
	$(CC) -o $@ $(CFLAGS_DEBUG) $< $(CODER_CUDA_OBJECTS_DEBUG)

PredictorNCUDA_d.o: staticPredictors-cuda/PredictorNCUDA.cu staticPredictors-cuda/PredictorNCUDA.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<

PredictorNWCUDA_d.o: staticPredictors-cuda/PredictorNWCUDA.cu staticPredictors-cuda/PredictorNWCUDA.h
	$(CC) -c -o $@ : error: ‘PGMCBP’ was not declared in this scope
	$(CFLAGS_DEBUG) $<

PredictorGWCUDA_d.o: staticPredictors-cuda/PredictorGWCUDA.cu staticPredictors-cuda/PredictorGWCUDA.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<

PredictorWCUDA_d.o: staticPredictors-cuda/PredictorWCUDA.cu staticPredictors-cuda/PredictorWCUDA.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<

PredictorNECUDA_d.o: staticPredictors-cuda/PredictorNECUDA.cu staticPredictors-cuda/PredictorNECUDA.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<

PredictorGNCUDA_d.o: staticPredictors-cuda/PredictorGNCUDA.cu staticPredictors-cuda/PredictorGNCUDA.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<

PredictorPLCUDA_d.o: staticPredictors-cuda/PredictorPLCUDA.cu staticPredictors-cuda/PredictorPLCUDA.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<

PGMCBPCCUDA_d.o: cbpc-cuda/PGMCBPCCUDA.cu cbpc-cuda/PGMCBPCCUDA.h config.h util.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<

$(CODER): cbpc/cbpc.cpp $(CODER_OBJECTS) config.h
	$(CC) -o $@ $(CFLAGS) $< $(CODER_OBJECTS)

PredictorN.o: staticPredictors/PredictorN.cpp staticPredictors/PredictorN.h
	$(CC) -c -o $@ $(CFLAGS) $<

PredictorNW.o: staticPredictors/PredictorNW.cpp staticPredictors/PredictorNW.h
	$(CC) -c -o $@ $(CFLAGS) $<

PredictorGW.o: staticPredictors/PredictorGW.cpp staticPredictors/PredictorGW.h
	$(CC) -c -o $@ $(CFLAGS) $<

PredictorW.o: staticPredictors/PredictorW.cpp staticPredictors/PredictorW.h
	$(CC) -c -o $@ $(CFLAGS) $<

PredictorNE.o: staticPredictors/PredictorNE.cpp staticPredictors/PredictorNE.h
	$(CC) -c -o $@ $(CFLAGS) $<

PredictorGN.o: staticPredictors/PredictorGN.cpp staticPredictors/PredictorGN.h
	$(CC) -c -o $@ $(CFLAGS) $<

PredictorPL.o: staticPredictors/PredictorPL.cpp staticPredictors/PredictorPL.h
	$(CC) -c -o $@ $(CFLAGS) $<

Predictor.o: staticPredictors/Predictor.cpp staticPredictors/Predictor.h
	$(CC) -c -o $@ $(CFLAGS) $<

PGMCBPC.o: cbpc/PGMCBPC.cpp cbpc/PGMCBPC.h config.h util.h
	$(CC) -c -o $@ $(CFLAGS) $<

$(CODER_DEBUG): cbpc/cbpc.cpp $(CODER_OBJECTS_DEBUG)
	$(CC) -o $@ $(CFLAGS_DEBUG) $< $(CODER_OBJECTS_DEBUG)

PredictorN_d.o: staticPredictors/PredictorN.cpp staticPredictors/PredictorN.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<

PredictorNW_d.o: staticPredictors/PredictorNW.cpp staticPredictors/PredictorNW.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<

PredictorGW_d.o: staticPredictors/PredictorGW.cpp staticPredictors/PredictorGW.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<

PredictorW_d.o: staticPredictors/PredictorW.cpp staticPredictors/PredictorW.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<

PredictorNE_d.o: staticPredictors/PredictorNE.cpp staticPredictors/PredictorNE.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<

PredictorGN_d.o: staticPredictors/PredictorGN.cpp staticPredictors/PredictorGN.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<

PredictorPL_d.o: staticPredictors/PredictorPL.cpp staticPredictors/PredictorPL.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<

Predictor_d.o: staticPredictors/Predictor.cpp staticPredictors/Predictor.h
	$(CC) -c -o $@ $(CFLAGS) $<

PGMCBPC_d.o: cbpc/PGMCBPC.cpp cbpc/PGMCBPC.h config.h util.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<


###########
# DECODER #
###########

$(DECODER_CUDA): cbpd-cuda/cbpd.cpp $(DECODER_CUDA_OBJECTS) config.h
	$(CC) -o $@ $(CFLAGS) $< $(DECODER_CUDA_OBJECTS)

PGMCBPDCUDA.o: cbpd-cuda/PGMCBPDCUDA.cu cbpd-cuda/PGMCBPDCUDA.h config.h util.h
	$(CC) -c -o $@ $(CFLAGS) $<

$(DECODER_CUDA_DEBUG): cbpd-cuda/cbpd.cpp $(DECODER_CUDA_OBJECTS_DEBUG) config.h
	$(CC) -o $@ $(CFLAGS_DEBUG) $< $(DECODER_CUDA_OBJECTS_DEBUG)

PGMCBPDCUDA_d.o: cbpd-cuda/PGMCBPDCUDA.cu cbpd-cuda/PGMCBPDCUDA.h config.h util.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<

$(DECODER): cbpd/cbpd.cpp $(DECODER_OBJECTS) config.h
	$(CC) -o $@ $(CFLAGS) $< $(DECODER_OBJECTS)

PGMCBPD.o: cbpd/PGMCBPD.cpp cbpd/PGMCBPD.h config.h util.h
	$(CC) -c -o $@ $(CFLAGS) $<

$(DECODER_DEBUG): cbpd/cbpd.cpp $(DECODER_OBJECTS_DEBUG) config.h
	$(CC) -o $@ $(CFLAGS_DEBUG) $< $(DECODER_OBJECTS_DEBUG)

PGMCBPD_d.o: cbpd/PGMCBPD.cpp cbpd/PGMCBPD.h config.h util.h
	$(CC) -c -o $@ $(CFLAGS_DEBUG) $<