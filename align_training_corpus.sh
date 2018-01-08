SOURCE=en
TARGET=ro
path_fast_align=/home/afm/fast_align/build
path_data=data/${SOURCE}-${TARGET}

paste -d '\t' \
      ${path_data}/corpus.bpe.sink.${SOURCE}.preprocessed \
      ${path_data}/corpus.bpe.${TARGET}.preprocessed \
      > ${path_data}/corpus.bpe.${SOURCE}-${TARGET}.preprocessed
sed -i 's/\t/ ||| /g' ${path_data}/corpus.bpe.${SOURCE}-${TARGET}.preprocessed
${path_fast_align}/fast_align -i ${path_data}/corpus.bpe.${SOURCE}-${TARGET}.preprocessed -d -o -v \
                  > ${path_data}/corpus.bpe.${SOURCE}-${TARGET}.preprocessed.forward.align
${path_fast_align}/fast_align -i ${path_data}/corpus.bpe.${SOURCE}-${TARGET}.preprocessed -d -o -v -r \
                  > ${path_data}/corpus.bpe.${SOURCE}-${TARGET}.preprocessed.reverse.align
