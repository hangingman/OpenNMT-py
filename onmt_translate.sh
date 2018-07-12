# Generic things
SOURCE=de
TARGET=en
LANGPAIR=${SOURCE}-${TARGET}
DATA=extra_data

# Specific things to translation
MODEL_NAME=de-en-md-base_acc_79.50_ppl_2.67_e15.pt
#MODEL_NAME=preprocessed_softmax_cattn-0_acc_43.02_ppl_69.09_e9.pt
SRC_FILE=test.de

# Call the OpenNMT-py script
python3 translate.py \
	    -model ${DATA}/${MODEL_NAME} \
		-src ${DATA}/${SRC_FILE} \
		-output ${DATA}/${SRC_FILE}.pred \
		-beam_size 10 \
        -min_length 2 \
		-use_guided \
		-tp_path "extra_data/translation_pieces_md_10-th0pt5.pickle"\
		-guided_n_max 4 \
		-guided_weight 1.0 \
        -guided_correct_ngrams \
        -guided_correct_1grams \
        -replace_unk
		#-length_penalty wu \
        #-alpha 1.5
        #-log_file "extra_data/log"

# Copy the predictions to the right folders
#HOME_PATH=/home/pmlf/Documents/github/NMT-Code/attention_comparison/thesis/guided_nmt
HOME_PATH=/home/ubuntu/NMT-Code/attention_comparison/thesis/guided_nmt
PRED_PATH=${HOME_PATH}/generate_results_de_en_domain/preds
MT_PATH=${HOME_PATH}/generate_results_de_en_domain/mt_predictions

#POS=guided_10-th0pt5-true
POS=base-true
#cp ${DATA}/${SRC_FILE}.pred ${PRED_PATH}/${SRC_FILE}.pred.${POS}
#sed -r 's/(@@ )|(@@ ?$)//g' ${PRED_PATH}/${SRC_FILE}.pred.${POS} > \
#	                                     ${MT_PATH}/${SRC_FILE}.pred.${POS}.merged
