SOURCE=de
TARGET=en
LANGPAIR=${SOURCE}-${TARGET}
DATA=/mnt/data/${LANGPAIR}-md
ONMT=/home/ubuntu/OpenNMT-py-fork
MODEL_FOLDER=base
MODEL_NAME=${LANGPAIR}-md-base

python3 ${ONMT}/train.py \
	-data ${DATA}/preprocessed-md \
	-save_model ${DATA}/${MODEL_FOLDER}/${MODEL_NAME} \
	-layers 2 \
	-encoder_type brnn \
	-rnn_size 512 \
	-dropout 0.3 \
	-epochs 15 \
	-seed 1 \
	-gpuid 0 \
	> ${DATA}/${MODEL_FOLDER}/${MODEL_NAME}.log 2> ${DATA}/${MODEL_FOLDER}/${MODEL_NAME}.err &

