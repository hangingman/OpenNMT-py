SOURCE=en
TARGET=ro
DATA_PATH=data/${SOURCE}-${TARGET}

for prefix in corpus newsdev2016 newstest2016
do
    sed 's/$/ <sink>/' ${DATA_PATH}/$prefix.bpe.${SOURCE} > ${DATA_PATH}/$prefix.bpe.sink.${SOURCE}
done

python preprocess.py \
       -train_src ${DATA_PATH}/corpus.bpe.sink.${SOURCE} \
       -train_tgt ${DATA_PATH}/corpus.bpe.${TARGET} \
       -valid_src ${DATA_PATH}/newsdev2016.bpe.sink.${SOURCE} \
       -valid_tgt ${DATA_PATH}/newsdev2016.bpe.${TARGET} \
       -write_txt \
       -save_data ${DATA_PATH}/preprocessed.sink.align
