echo "Initial Time:"
date +"%F %T,%3N"

# Generic things
SOURCE=de
TARGET=en
LANGPAIR=${SOURCE}-${TARGET}
save=true
POS=sparsemax_covvector

# German to English in-domain models
MODEL_PREFIX=/mnt/data/${LANGPAIR}-scnmt


# --------------------------- BASELINE MODELS ------------------------
#MODEL_NAME=${MODEL_PREFIX}/softmax/used/preprocessed_softmax_cattn-0_acc_42.90_ppl_68.61_e9.pt
#MODEL_NAME=${MODEL_PREFIX}/softmax/used/preprocessed_softmax_cattn-0_coverage-1_lambda-00_acc_42.62_ppl_69.82_e9.pt
#MODEL_NAME=${MODEL_PREFIX}/sparsemax/used/preprocessed_sparsemax_cattn-0_acc_42.98_ppl_66.92_e9.pt
MODEL_NAME=${MODEL_PREFIX}/sparsemax/used/preprocessed_sparsemax_cattn-0_coverage-1_lambda-00_acc_43.23_ppl_69.97_e10.pt

# --------------------------- CSOFTMAX MODELS ------------------------
#MODEL_NAME=${MODEL_PREFIX}/csoftmax/used/preprocessed_csoftmax_fixed-1.5_cattn-0_acc_42.52_ppl_70.01_e9.pt
#MODEL_NAME=${MODEL_PREFIX}/csoftmax/used/preprocessed_csoftmax_fixed-1.5_cattn-0.2_acc_42.64_ppl_70.71_e9.pt
#MODEL_NAME=${MODEL_PREFIX}/csoftmax/used/preprocessed_csoftmax_fixed-1.5_cattn-0.4_acc_42.50_ppl_70.46_e9.pt
#MODEL_NAME=${MODEL_PREFIX}/csoftmax/used/preprocessed_csoftmax_fixed-2_cattn-0.0_acc_42.87_ppl_67.23_e9.pt
#MODEL_NAME=${MODEL_PREFIX}/csoftmax/used/preprocessed_csoftmax_fixed-3_cattn-0_acc_42.66_ppl_68.30_e9.pt
#MODEL_NAME=${MODEL_PREFIX}/csoftmax/used/preprocessed_csoftmax_guided_cattn-0.0_acc_42.89_ppl_66.44_e9.pt
#MODEL_NAME=${MODEL_PREFIX}/csoftmax/used/preprocessed_csoftmax_actual_cattn-0.0_acc_42.91_ppl_68.39_e9.pt
#MODEL_NAME=${MODEL_PREFIX}/csoftmax/used/preprocessed_csoftmax_actual_cattn-0.2_acc_42.67_ppl_66.60_e9.pt
#MODEL_NAME=${MODEL_PREFIX}/csoftmax/used/preprocessed_csoftmax_actual_cattn-0.4_acc_42.88_ppl_67.89_e9.pt

# --------------------------- CSPARSEMAX MODELS ------------------------
#MODEL_NAME=${MODEL_PREFIX}/csparsemax/used/preprocessed_csparsemax_fixed-1.5_cattn-0_acc_42.80_ppl_71.71_e10.pt
#MODEL_NAME=${MODEL_PREFIX}/csparsemax/used/preprocessed_csparsemax_fixed-2_cattn-0.0_acc_43.64_ppl_68.81_e10.pt
#MODEL_NAME=${MODEL_PREFIX}/csparsemax/used/preprocessed_csparsemax_fixed-3_cattn-0.0_acc_43.70_ppl_69.74_e10.pt
#MODEL_NAME=${MODEL_PREFIX}/csparsemax/used/preprocessed_csparsemax_guided_cattn-0.0_acc_43.88_ppl_68.62_e11.pt
#MODEL_NAME=${MODEL_PREFIX}/csparsemax/used/preprocessed_csparsemax_actual_cattn-0.0_acc_43.18_ppl_71.96_e9.pt
#MODEL_NAME=${MODEL_PREFIX}/csparsemax/used/preprocessed_csparsemax_actual_cattn-0.2_acc_43.25_ppl_66.24_e9.pt
#MODEL_NAME=${MODEL_PREFIX}/csparsemax/used/preprocessed_csparsemax_actual_cattn-0.4_acc_43.69_ppl_69.30_e10.pt

# Specify the source file
SRC_FILE=/mnt/data/${LANGPAIR}-scnmt/test.bpe.sink.${SOURCE}
#SRC_FILE=/mnt/data/${LANGPAIR}-scnmt/dev.bpe.sink.${SOURCE}

for alpha in 0.0 ; do
    for beta in 0.0 ; do

        #
        echo "Using: "
        echo "${MODEL_NAME}"

        # Call the OpenNMT-py script
        python translate.py \
                -model ${MODEL_NAME} \
                -src ${SRC_FILE} \
                -output ${SRC_FILE}.pred \
                -beam_size 5 \
                -batch_size 1 \
                -min_length 3 \
                -min_attention 0.1 \
                -coverage_penalty "wu" \
                -length_penalty "wu" \
                -alpha ${alpha} \
                -beta ${beta} \
                -replace_unk \
                -gpu 0
        # CHANGE THE NAME OF THE FILE
        #v11=$(echo ${alpha} | cut -f1 -d.)
        #v12=$(echo ${alpha} | cut -f2 -d.)
        #v21=$(echo ${beta} | cut -f1 -d.)
        #v22=$(echo ${beta} | cut -f2 -d.)

        #POS=sparsemax_covpenalty-${v11}pt${v12}-${v21}pt${v22}

        # Copy the predictions to the right folders
        HOME_PATH=/home/ubuntu/NMT-Code/attention_comparison/thesis/constrained_sparse_experiments
        PRED_PATH=${HOME_PATH}/generate_results_de_en/preds
        MT_PATH=${HOME_PATH}/generate_results_de_en/mt_predictions

        if ${save}; then
            FN=$(echo ${SRC_FILE} | cut -d'/' -f5)
            cp ${SRC_FILE}.pred ${PRED_PATH}/${FN}.pred.${POS}
            sed -r 's/(@@ )|(@@ ?$)//g' ${PRED_PATH}/${FN}.pred.${POS} > \
                                        ${MT_PATH}/${FN}.pred.${POS}.merged
        fi

        echo "Finishing Time:"
        date +"%F %T,%3N"
    done
done
