# Generic things
SOURCE=de
TARGET=en
LANGPAIR=${SOURCE}-${TARGET}

# German to English in-domain models
#MODEL_NAME=/mnt/data/${LANGPAIR}-scnmt/csoftmax_predicted__0.8/preprocessed_csoftmax_predicted_cattn-0.8_acc_43.81_ppl_68.65_e13.pt
MODEL_NAME=/mnt/data/${LANGPAIR}-scnmt/csparsemax_predicted__0.8/preprocessed_csparsemax_predicted_cattn-0.8_acc_28.38_ppl_221.98_e1.pt

# Specify the source file
SRC_FILE=/mnt/data/${LANGPAIR}-scnmt/test.bpe.sink.${SOURCE}

# Call the OpenNMT-py script
python translate.py \
        -model ${MODEL_NAME} \
        -src ${SRC_FILE} \
        -output ${SRC_FILE}.pred \
        -beam_size 5 \
        -batch_size 1 \
        -min_attention 0.1 \
        -replace_unk \
        -gpu 0
# CHANGE THE NAME OF THE FILE

# Copy the predictions to the right folders
HOME_PATH=/home/ubuntu/NMT-Code/attention_comparison/thesis/guided_nmt
PRED_PATH=${HOME_PATH}/generate_results_de_enX/preds
MT_PATH=${HOME_PATH}/generate_results_de_enX/mt_predictions

#POS=base

#FN=$(echo ${SRC_FILE} | cut -d'/' -f5)
#cp ${SRC_FILE}.pred ${PRED_PATH}/${FN}.pred.${POS}
#sed -r 's/(@@ )|(@@ ?$)//g' ${PRED_PATH}/${FN}.pred.${POS} > \
#                            ${MT_PATH}/${FN}.pred.${POS}.merged


date +"%F %T,%3N"
