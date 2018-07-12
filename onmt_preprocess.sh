SOURCE=de
TARGET=en
LANGPAIR=${SOURCE}-${TARGET}
DATA=/mnt/data/${LANGPAIR}-md-shr
ONMT=/home/ubuntu/OpenNMT-py-un

python -u ${ONMT}/preprocess.py \
	-train_src ${DATA}/train.bpe.sink.${SOURCE} \
	-train_tgt ${DATA}/train.bpe.${TARGET} \
	-valid_src ${DATA}/dev.bpe.sink.${SOURCE} \
	-valid_tgt ${DATA}/dev.bpe.${TARGET} \
	-save_data ${DATA}/preprocessed-md-shr

