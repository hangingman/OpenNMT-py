gpu=$1
SOURCE=$2 # ro
TARGET=$3 # en
LANGPAIR=${SOURCE}-${TARGET}
#DATA=/mnt/data/home/afm/mt_data/data/${LANGPAIR}
#MODEL=/mnt/data/home/afm/mt_data/model/${LANGPAIR}
DATA=/mnt/disk/afm/data/${LANGPAIR}
MODEL=/mnt/disk/afm/model/${LANGPAIR}
LOGS=logs
ATTN=$4 # softmax|sparsemax|csoftmax|csparsemax
cattn=$5 # 0|0.2
FERTTYPE=$6 # none|fixed|guided
FERTILITY=$7 # none|2|3
train=true

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
mkdir -p ${LOGS}

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
               -fertility_type fixed \
               -c_attn ${cattn} -seed 42 -gpuid ${gpu} &> \
               ${LOGS}/log_${LANGPAIR}_${ATTN}_${FERTTYPE}-${FERTILITY}_cattn-${cattn}.txt
    elif [ "$FERTTYPE" == "guided" ]
    then
        python -u train.py -data ${DATA}/preprocessed.sink.align \
               -save_model ${MODEL}/preprocessed_${ATTN}_${FERTTYPE}_cattn-${cattn} \
               -attn_transform ${TRANSFORM} \
               -fertility_type guided \
               -c_attn ${cattn} -seed 42 -gpuid ${gpu} &> \
               ${LOGS}/log_${LANGPAIR}_${ATTN}_${FERTTYPE}_cattn-${cattn}.txt
    fi
fi
