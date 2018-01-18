gpu=$1
SOURCE=$2 # ro
TARGET=$3 # en
LANGPAIR=${SOURCE}-${TARGET}
DATA=/mnt/data/home/afm/mt_data/data/${LANGPAIR}
MODEL=/mnt/data/home/afm/mt_data/model/${LANGPAIR}
LOGS=logs
ATTN=$4 # softmax|sparsemax|csoftmax|csparsemax
cattn=$5 # 0|0.2
FERTTYPE=$6 # none|fixed|guided
FERTILITY=$7 # none|2|3
train=true
preprocess=false

if [ "$ATTN" == "csparsemax" ]
then
    TRANSFORM=constrained_sparsemax
elif [ "$ATTN" == "csoftmax" ]
then
    TRANSFORM=constrained_softmax
else
    TRANSFORM=${ATTN}
fi

cd ..
mkdir -p ${MODEL}

if $preprocess
then
    for prefix in corpus newsdev2016 newstest2016
    do
        sed 's/$/ <sink>/' ${DATA}/$prefix.bpe.${SOURCE} > ${DATA}/$prefix.bpe.sink.${SOURCE}
    done

    python preprocess.py \
           -train_src ${DATA}/corpus.bpe.sink.${SOURCE} \
           -train_tgt ${DATA}/corpus.bpe.${TARGET} \
           -valid_src ${DATA}/newsdev2016.bpe.sink.${SOURCE} \
           -valid_tgt ${DATA}/newsdev2016.bpe.${TARGET} \
           -save_data ${DATA}/preprocessed.sink.align # \
           #-write_txt

    if false
    then
    ALIGNER=/home/afm/fast_align/build

    paste -d '\t' \
          ${DATA}/corpus.bpe.sink.${SOURCE}.preprocessed \
          ${DATA}/corpus.bpe.${TARGET}.preprocessed \
          > ${DATA}/corpus.bpe.${SOURCE}-${TARGET}.preprocessed
    sed -i 's/\t/ ||| /g' ${DATA}/corpus.bpe.${SOURCE}-${TARGET}.preprocessed
    ${ALIGNER}/fast_align -i ${DATA}/corpus.bpe.${SOURCE}-${TARGET}.preprocessed -d -o -v \
              > ${DATA}/corpus.bpe.${SOURCE}-${TARGET}.preprocessed.forward.align
    ${ALIGNER}/fast_align -i ${DATA}/corpus.bpe.${SOURCE}-${TARGET}.preprocessed -d -o -v -r \
              > ${DATA}/corpus.bpe.${SOURCE}-${TARGET}.preprocessed.reverse.align
    fi
fi

if $train
then
    if [ "$ATTN" == "softmax" ] || [ "$ATTN" == "sparsemax" ]
    then
        python -u train.py -data ${DATA}/preprocessed.sink.align \
               -save_model ${MODEL}/preprocessed_${ATTN}_cattn-${cattn} \
               -attn_transform ${TRANSFORM} \
               -c_attn ${cattn} -seed 42 -gpuid ${gpu} &> \
               ${LOGS}/log_${LANGPAIR}_${ATTN}_cattn-${cattn}.txt
    elif [ "$FERTTYPE" == "fixed" ]
    then
        python -u train.py -data ${DATA}/preprocessed.sink.align \
               -save_model ${MODEL}/preprocessed_${ATTN}_${FERTTYPE}-${FERTILITY}_cattn-${cattn} \
               -attn_transform ${TRANSFORM} \
               -fertility ${FERTILITY} \
               -c_attn ${cattn} -seed 42 -gpuid ${gpu} &> \
               ${LOGS}/log_${LANGPAIR}_${ATTN}_${FERTTYPE}-${FERTILITY}_cattn-${cattn}.txt
    elif [ "$FERTTYPE" == "guided" ]
    then
        python -u train.py -data ${DATA}/preprocessed.sink.align \
               -save_model ${MODEL}/preprocessed_${ATTN}_${FERTTYPE}_cattn-${cattn} \
               -attn_transform ${TRANSFORM} \
               -guided_fertility ${DATA}/corpus.bpe.${LANGPAIR}.preprocessed.forward.align \
               -guided_fertility_source_file ${DATA}/corpus.bpe.sink.${SOURCE}.preprocessed \
               -c_attn ${cattn} -seed 42 -gpuid ${gpu} &> \
               ${LOGS}/log_${LANGPAIR}_${ATTN}_${FERTTYPE}_cattn-${cattn}.txt
    fi
fi
