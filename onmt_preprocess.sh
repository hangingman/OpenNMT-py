SOURCE=es
TARGET=en
LANGPAIR=${SOURCE}-${TARGET}
DATA=/mnt/data/${LANGPAIR}-shr-big
ONMT=/home/ubuntu/OpenNMT-py-un

python3 ${ONMT}/preprocess.py \
	-train_src ${DATA}/train.bpe.${SOURCE} \
	-train_tgt ${DATA}/train.bpe.${TARGET} \
	-valid_src ${DATA}/dev.bpe.${SOURCE} \
	-valid_tgt ${DATA}/dev.bpe.${TARGET} \
	-save_data ${DATA}/preprocessed-${LANGPAIR}-shr-big

