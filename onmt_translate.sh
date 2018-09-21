# Generic things
SOURCE=de
TARGET=en
LANGPAIR=${SOURCE}-${TARGET}

# Specific things to translation
#MODEL_NAME=de-en-shr-base_acc_43.15_ppl_63.81_e9.pt
#MODEL_NAME=/mnt/data/${LANGPAIR}-shr/base/de-en-md-shr_acc_79.67_ppl_2.58_e15.pt
#MODEL_NAME=/mnt/data/${LANGPAIR}-shr/base/en-de-jrc-shr_acc_77.19_ppl_3.02_e15.pt
#MODEL_NAME=/mnt/data/${LANGPAIR}-shr-big/base/de-en-shr-big_acc_55.25_ppl_12.96_e15.pt
#MODEL_NAME=/mnt/data/${LANGPAIR}-md-shr-small-da-genvocab/5k/gen_indomain_5k_acc_47.22_ppl_36.32_e25.pt
#MODEL_NAME=/mnt/data/${LANGPAIR}-md-shr-small-da-genvocab/10k/gen_indomain_10k_acc_48.82_ppl_32.45_e25.pt
#MODEL_NAME=/mnt/data/${LANGPAIR}-md-shr-small-da-genvocab/15k/gen_indomain_15k_acc_49.84_ppl_29.81_e28.pt
#MODEL_NAME=/mnt/data/${LANGPAIR}-md-shr-small-da-genvocab/20k/gen_indomain_20k_acc_50.55_ppl_27.94_e25.pt
#MODEL_NAME=/mnt/data/${LANGPAIR}-md-shr-small-da-genvocab/30k/gen_indomain_30k_acc_51.45_ppl_25.25_e26.pt
MODEL_NAME=/mnt/data/de-en-scnmt/csoftmax_predicted__0.8/preprocessed_csoftmax_predicted_cattn-0.8_acc_43.39_ppl_67.42_e10.pt

# Specify the source file
#SRC_FILE=/mnt/data/${LANGPAIR}-md-shr-small-da-genvocab/test.bpe.sink.${SOURCE}
SRC_FILE=/mnt/data/${LANGPAIR}-scnmt/test.bpe.sink.${SOURCE}

# Specify TP path
TP_PATH=/mnt/translation_pieces/${LANGPAIR}-md-genvocab

#for lambda1 in 3.5
#do
#    for extrald1 in 5.0
#    do
 
#        if [ $(echo "${lambda1}==${lambda2}"|bc) -eq 1 ]
#        then
#            echo "Skipping ${lambda1} and ${lambda2}"
#            continue
#        fi

        lambda1=3.5
        lambda2=4.0
        extrald1=5.0
        extrald2=5.0
        m=20
        th=3
        suffix=1000000000000000000
        fntmodel=30

        echo "Lambda1: ${lambda1}"
        echo "Lambda2: ${lambda2}"
        echo "Extrald1: ${extrald1}"
        echo "Extrald2: ${extrald2}"
        echo "M: ${m}" 
        echo "Threshold: 0.${th}"
        echo "Finetuning: ${fntmodel}"
        echo "Extra tps: ${suffix}"
 
        # Call the OpenNMT-py script
        python3 translate.py \
                -model ${MODEL_NAME} \
                -src ${SRC_FILE} \
                -output ${SRC_FILE}.pred \
                -beam_size 5 \
                -min_length 3 \
                -fertility 2 \
                -fertility_type "fixed" \
                -replace_unk \
                -gpu 0
                #-use_guided \
                #-tp_path ${TP_PATH}/test_translation_pieces_md_${m}-th0pt${th}_finetuning_${suffix}k.pickle \
                #-guided_n_max 4 \
                #-guided_1_weight ${lambda1} \
                #-guided_n_weight ${lambda2} \
                #-guided_correct_ngrams \
                #-guided_correct_1grams \
                #-extend_with_tp \
                #-extend_1_weight ${extrald1} \
                #-extend_n_weight ${extrald2} \
                #-replace_unk

        # CHANGE THE NAME OF THE FILE
    
        # Copy the predictions to the right folders
        HOME_PATH=/home/ubuntu/NMT-Code/attention_comparison/thesis/guided_nmt
        PRED_PATH=${HOME_PATH}/generate_results_de_en_da/preds
        MT_PATH=${HOME_PATH}/generate_results_de_en_da/mt_predictions

        v11=$(echo ${lambda1} | cut -f1 -d.)
        v12=$(echo ${lambda1} | cut -f2 -d.)
        v21=$(echo ${lambda2} | cut -f1 -d.)
        v22=$(echo ${lambda2} | cut -f2 -d.)
        
        v31=$(echo ${extrald1} | cut -f1 -d.)
        v32=$(echo ${extrald1} | cut -f2 -d.)
        v41=$(echo ${extrald2} | cut -f1 -d.)
        v42=$(echo ${extrald2} | cut -f2 -d.)
        
        #POS=guided-${m}-0pt${th}-lambda-${v11}pt${v12}-${v21}pt${v22}-extrald-${v31}pt${v32}-${v41}pt${v42}-dynamic_unk-mod45-finetuned_${fntmodel}k-extra_${suffix}k
        #POS=guided-${m}-0pt${th}-lambda-${v11}pt${v12}-${v21}pt${v22}-extrald-${v31}pt${v32}-${v41}pt${v42}-dynamic_unk-mod45-finetuned_${fntmodel}k
        #POS=guided-${m}-0pt${th}-lambda-${v11}pt${v12}-${v21}pt${v22}-extrald-${v31}pt${v32}-${v41}pt${v42}-dynamic_unk-mod45-extra_${suffix}k
        #POS=guided-10-0pt0-lambda-${v11}pt${v12}-${v21}pt${v22}
        #POS=teste
        
        #FN=$(echo ${SRC_FILE} | cut -d'/' -f5)
        #cp ${SRC_FILE}.pred ${PRED_PATH}/${FN}.pred.${POS}
        #sed -r 's/(@@ )|(@@ ?$)//g' ${PRED_PATH}/${FN}.pred.${POS} > \
        #                            ${MT_PATH}/${FN}.pred.${POS}.merged

#    done
#done

date +"%F %T,%3N"
