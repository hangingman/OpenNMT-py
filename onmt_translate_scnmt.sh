echo "Initial Time:"
date +"%F %T,%3N"

# Generic things
SOURCE=de
TARGET=en
LANGPAIR=${SOURCE}-${TARGET}

# German to English in-domain models
MODEL_PREFIX=/mnt/data/${LANGPAIR}-scnmt
#MODEL_NAME=${MODEL_PREFIX}/csoftmax/used/preprocessed_csoftmax_fixed-2_cattn-0.0_acc_42.87_ppl_67.23_e9.pt
MODEL_NAME=${MODEL_PREFIX}/softmax/preprocessed_softmax_cattn-0_acc_42.90_ppl_68.61_e9.pt
#MODEL_NAME=${MODEL_PREFIX}/sparsemax/preprocessed_sparsemax_cattn-0_acc_42.98_ppl_66.92_e9.pt
#MODEL_NAME=${MODEL_PREFIX}/csparsemax/used/preprocessed_csparsemax_actual_cattn-0.0_acc_43.18_ppl_71.96_e9.pt
#MODEL_NAME=${MODEL_PREFIX}/csparsemax/used/preprocessed_csparsemax_fixed-2_cattn-0.0_acc_43.64_ppl_68.81_e10.pt
#MODEL_NAME=${MODEL_PREFIX}/csparsemax/used/preprocessed_csparsemax_fixed-3_cattn-0.0_acc_43.70_ppl_69.74_e10.pt
#MODEL_NAME=${MODEL_PREFIX}/csparsemax/used/preprocessed_csparsemax_guided_cattn-0.0_acc_43.88_ppl_68.62_e11.pt


# Specify the source file
#SRC_FILE=/mnt/data/${LANGPAIR}-scnmt/test.bpe.sink.${SOURCE}
SRC_FILE=/mnt/data/${LANGPAIR}-scnmt/dev.bpe.sink.${SOURCE}

# Call the OpenNMT-py script
python translate.py \
        -model ${MODEL_NAME} \
        -src ${SRC_FILE} \
        -output ${SRC_FILE}.pred \
        -beam_size 5 \
        -batch_size 1 \
        -min_length 3 \
        -min_attention 0.1 \
        -replace_unk \
        -gpu 0
# CHANGE THE NAME OF THE FILE

# Copy the predictions to the right folders
HOME_PATH=/home/ubuntu/NMT-Code/attention_comparison/thesis/constrained_sparse_experiments
PRED_PATH=${HOME_PATH}/generate_results_de_en/preds
MT_PATH=${HOME_PATH}/generate_results_de_en/mt_predictions

POS=softmax

FN=$(echo ${SRC_FILE} | cut -d'/' -f5)
cp ${SRC_FILE}.pred ${PRED_PATH}/${FN}.pred.${POS}
sed -r 's/(@@ )|(@@ ?$)//g' ${PRED_PATH}/${FN}.pred.${POS} > \
                            ${MT_PATH}/${FN}.pred.${POS}.merged

echo "Finishing Time:"
date +"%F %T,%3N"
