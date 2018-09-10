PREFIX=$1
FOLDER=/mnt/data/de-en-md-shr-small-da

# 1. Preprocess the new temporary data
rm ${FOLDER}/gen_indomain.*

python3 preprocess.py -train_src ${FOLDER}/temp.bpe.sink.de \
                      -train_tgt ${FOLDER}/temp.bpe.en \
                      -valid_src ${FOLDER}/dev.bpe.sink.de \
                      -valid_tgt ${FOLDER}/dev.bpe.en \
                      -save_data ${FOLDER}/gen_indomain

# 2. Train the in-domain
python train.py -data ${FOLDER}/gen_indomain \
                -train_from ${FOLDER}/de-en-shr-big_acc_55.25_ppl_12.96_e15.pt \
                -save_model ${FOLDER}/${PREFIX}/gen_indomain_${PREFIX} \
                -layers 2 \
                -encoder_type brnn \
                -dropout 0.3 \
                -share_decoder_embeddings \
                -epochs 20 \
                -seed 42 \
                -gpu 0 \
                > ${FOLDER}/${PREFIX}/gen_indomain_${PREFIX}.log \
                2> ${FOLDER}/${PREFIX}/gen_indomain_${PREFIX}.err

