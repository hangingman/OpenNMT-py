SOURCE=de
TARGET=en
LANGPAIR=${SOURCE}-${TARGET}
DATA=/mnt/data/${LANGPAIR}-md-shr-small-da-genvocab
ONMT=/home/ubuntu/OpenNMT-py-un

python3 ${ONMT}/preprocess.py \
	-train_src ${DATA}/temp.bpe.sink.${SOURCE} \
	-train_tgt ${DATA}/temp.bpe.${TARGET} \
	-valid_src ${DATA}/dev.bpe.sink.${SOURCE} \
	-valid_tgt ${DATA}/dev.bpe.${TARGET} \
	-save_data ${DATA}/gen-indomain

