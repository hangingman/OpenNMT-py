# Generic things
SOURCE=es
TARGET=en
LANGPAIR=${SOURCE}-${TARGET}

# German to English in-domain models
#MODEL_NAME=/mnt/data/${LANGPAIR}-md-shr/base/de-en-md-shr_acc_79.67_ppl_2.58_e15.pt
#MODEL_NAME=/mnt/data/${LANGPAIR}-shr-big/base/de-en-shr-big_acc_55.25_ppl_12.96_e15.pt

# Spanish to English in-domain models
MODEL_NAME=/mnt/data/${LANGPAIR}-md-shr/base/es-en-md-shr_acc_74.58_ppl_4.06_e14.pt
#MODEL_NAME=/mnt/data/${LANGPAIR}-shr-big/base/es-en-shr-big_acc_68.32_ppl_4.82_e15.pt

# Specify the source file
SRC_FILE=/mnt/data/${LANGPAIR}-md-shr/test.bpe.${SOURCE}

# Specify TP path
TP_PATH=/mnt/translation_pieces/${LANGPAIR}-md

# Hyperparameters
lambda1=1.1
lambda2=1.0
extrald1=0.0
extrald2=${extrald1}
m=5
th=0

echo "Lambda1: ${lambda1}"
echo "Lambda2: ${lambda2}"
echo "Extrald1: ${extrald1}"
echo "Extrald2: ${extrald2}"
echo "M: ${m}" 
echo "Threshold: 0.${th}"

# Call the OpenNMT-py script
python3 translate.py \
        -model ${MODEL_NAME} \
        -src ${SRC_FILE} \
        -output ${SRC_FILE}.pred \
        -beam_size 5 \
        -min_length 3 \
        -use_guided \
        -tp_path ${TP_PATH}/test_translation_pieces_md_${m}-th0pt${th}.pickle-high \
        -guided_n_max 4 \
        -guided_1_weight ${lambda1} \
        -guided_n_weight ${lambda2} \
        -guided_correct_ngrams \
        -guided_correct_1grams \
        -replace_unk
        #-extend_with_tp \
        #-extend_1_weight ${extrald1} \
        #-extend_n_weight ${extrald2} \
        #-replace_unk
# CHANGE THE NAME OF THE FILE

# Copy the predictions to the right folders
HOME_PATH=/home/ubuntu/NMT-Code/attention_comparison/thesis/guided_nmt
PRED_PATH=${HOME_PATH}/generate_results_es_en_md/preds
MT_PATH=${HOME_PATH}/generate_results_es_en_md/mt_predictions

v11=$(echo ${lambda1} | cut -f1 -d.)
v12=$(echo ${lambda1} | cut -f2 -d.)
v21=$(echo ${lambda2} | cut -f1 -d.)
v22=$(echo ${lambda2} | cut -f2 -d.)

v31=$(echo ${extrald1} | cut -f1 -d.)
v32=$(echo ${extrald1} | cut -f2 -d.)
v41=$(echo ${extrald2} | cut -f1 -d.)
v42=$(echo ${extrald2} | cut -f2 -d.)

POS=guided-${m}-0pt${th}-lambda-${v11}pt${v12}-${v21}pt${v22}-extrald-${v31}pt${v32}-${v41}pt${v42}-dynamic_unk-high
#POS=guided-${m}-0pt${th}-lambda-${v11}pt${v12}-${v21}pt${v22}
#POS=base

FN=$(echo ${SRC_FILE} | cut -d'/' -f5)
cp ${SRC_FILE}.pred ${PRED_PATH}/${FN}.pred.${POS}
sed -r 's/(@@ )|(@@ ?$)//g' ${PRED_PATH}/${FN}.pred.${POS} > \
                            ${MT_PATH}/${FN}.pred.${POS}.merged


date +"%F %T,%3N"
