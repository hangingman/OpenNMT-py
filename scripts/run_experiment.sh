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
preprocess=false #true

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
           -write_txt \
           -save_data ${DATA}/preprocessed.sink.align

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

if $train
then
    if [ "$ATTN" == "softmax" ] || [ "$ATTN" == "sparsemax" ]
    then
        python -u train.py -data ${DATA}/preprocessed.sink.align.train.pt \
               -save_model ${MODEL}/preprocessed_${ATTN}_cattn-${cattn} \
               -attn_transform ${TRANSFORM} \
               -start_epoch 7 -train_from ${MODEL}/preprocessed_sparsemax_cattn-0_acc_39.22_ppl_48.04_e6.pt \
               -c_attn ${cattn} -seed 42 -gpus ${gpu} &> \
               ${LOGS}/log_${LANGPAIR}_${ATTN}_cattn-${cattn}.txt
    elif [ "$FERTTYPE" == "fixed" ]
    then
        python -u train.py -data ${DATA}/preprocessed.sink.align.train.pt \
               -save_model ${MODEL}/preprocessed_${ATTN}_${FERTTYPE}-${FERTILITYT}_cattn-${cattn} \
               -attn_transform ${TRANSFORM} \
               -fertility ${FERTILITY} \
               -c_attn ${cattn} -seed 42 -gpus ${gpu} &> \
               ${LOGS}/log_${LANGPAIR}_${ATTN}_${FERTTYPE}-${FERTILITY}_cattn-${cattn}.txt
    elif [ "$FERTTYPE" == "guided" ]
    then
        python -u train.py -data ${DATA}/preprocessed.sink.align.train.pt \
               -save_model ${MODEL}/preprocessed_${ATTN}_${FERTTYPE}_cattn-${cattn} \
               -attn_transform ${TRANSFORM} \
               -guided_fertility ${DATA}/corpus.bpe.${LANGPAIR}.preprocessed.forward.align \
               -guided_fertility_source_file ${DATA}/corpus.bpe.sink.${SOURCE}.preprocessed \
               -start_epoch 4 -train_from ${MODEL}/preprocessed_csoftmax_guided_cattn-0.2_acc_37.88_ppl_58.60_e3.pt \
               -c_attn ${cattn} -seed 42 -gpus ${gpu} &> \
               ${LOGS}/log_${LANGPAIR}_${ATTN}_${FERTTYPE}_cattn-${cattn}.txt
    fi
fi
