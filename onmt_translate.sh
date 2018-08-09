# Generic things
SOURCE=en
TARGET=de
LANGPAIR=${SOURCE}-${TARGET}
DATA=extra_data

# Specific things to translation
#MODEL_NAME=de-en-shr-base_acc_43.15_ppl_63.81_e9.pt
#MODEL_NAME=de-en-md-shr_acc_79.77_ppl_2.60_e15.pt
#MODEL_NAME=de-en-md-base_acc_79.50_ppl_2.67_e15.pt
#MODEL_NAME=preprocessed_softmax_cattn-0_acc_43.02_ppl_69.09_e9.pt
MODEL_NAME=en-de-jrc-shr_acc_77.19_ppl_3.02_e15.pt

# Specify the source file
#SRC_FILE=test.bpe.sink.de
SRC_FILE=dev.bpe.sink.en

#for a in 0.4 0.6 
#do
#    for b in 0.1 0.2
#    do
      
#        echo "Alpha: ${a}"
#        echo "Beta:  ${b}"

#for lambda1 in 0.9 1 1.2 1.3
#do
#    for lambda2 in 0.9 1 1.2 1.5
#    do
 
#        if [ $(echo "${lambda1}==${lambda2}"|bc) -eq 1 ]
#        then
#            echo "Skipping ${lambda1} and ${lambda2}"
#            continue
#        fi
 
#        echo "Lambda1: ${lambda1}"
#        echo "Lambda2: ${lambda2}"
 
        # Call the OpenNMT-py script
        python3 translate.py \
                -model ${DATA}/models/${MODEL_NAME} \
                -src ${DATA}/data/en-de-jrc/${SRC_FILE} \
                -output ${DATA}/${SRC_FILE}.pred \
                -beam_size 5 \
                -min_length 2 \
                -use_guided \
                -tp_path "extra_data/t_pieces/en-de-jrc/dev_translation_pieces_md_10-th0pt0.pickle" \
                -guided_n_max 4 \
                -guided_1_weight 1.5 \
                -guided_n_weight 1.5 \
                -replace_unk
                #-guided_correct_ngrams \
                #-guided_correct_1grams \
                #-replace_unk
                #-length_penalty wu \
                #-coverage_penalty wu \
                #-alpha ${a} \
                #-beta ${b} \
                #-replace_unk
                #-gpu 0

        # Copy the predictions to the right folders
        #HOME_PATH=/home/pmlf/Documents/github/NMT-Code/attention_comparison/thesis/guided_nmt
        HOME_PATH=/home/ubuntu/NMT-Code/attention_comparison/thesis/guided_nmt

        PRED_PATH=${HOME_PATH}/generate_results_en_de_jrc/preds
        MT_PATH=${HOME_PATH}/generate_results_en_de_jrc/mt_predictions

        #v11=$(echo ${a} | cut -f1 -d.)
        #v12=$(echo ${a} | cut -f2 -d.)
        #v21=$(echo ${b} | cut -f1 -d.)
        #v22=$(echo ${b} | cut -f2 -d.)
        #POS=wu-${v11}pt${v12}-${v21}pt${v22}

#        v11=$(echo ${lambda1} | cut -f1 -d.)
#        v12=$(echo ${lambda1} | cut -f2 -d.)
#        v21=$(echo ${lambda2} | cut -f1 -d.)
#        v22=$(echo ${lambda2} | cut -f2 -d.)
#        POS=5-0pt0-rem-${v11}pt${v12}-${v21}pt${v22}
        POS=paper

        cp ${DATA}/${SRC_FILE}.pred ${PRED_PATH}/${SRC_FILE}.pred.${POS}
        sed -r 's/(@@ )|(@@ ?$)//g' ${PRED_PATH}/${SRC_FILE}.pred.${POS} > \
                                     ${MT_PATH}/${SRC_FILE}.pred.${POS}.merged

#    done
#done
