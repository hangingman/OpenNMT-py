SOURCE=de
TARGET=en
LANGPAIR=${SOURCE}-${TARGET}
DATA=/mnt/data/${LANGPAIR}-md
ONMT=/home/ubuntu/OpenNMT-py-fork

python3 ${ONMT}/preprocess.py \
	-train_src ${DATA}/extra.${SOURCE} \
	-train_tgt ${DATA}/extra.${TARGET} \
	-valid_src ${DATA}/dev.${SOURCE} \
	-valid_tgt ${DATA}/dev.${TARGET} \
	-save_data ${DATA}/preprocessed-md

