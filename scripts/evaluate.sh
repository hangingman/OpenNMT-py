model=$1
source=$2
target=$3

srclang=en #ro #ro_small
tgtlang=ro #en
langpair=${srclang}-${tgtlang}
align=data/${langpair}/corpus.bpe.${langpair}.forward.align
train_src=data/${langpair}/corpus.bpe.${srclang}
alpha=0 #0.2
beta=0 #0.2
#c_attn=0 #0.2

#python translate.py -model $model -src $source -output $target.pred -attn_transform constrained_softmax -guided_fertility $align -guided_fertility_source_file ${train_src} -beam_size 10 -alpha ${alpha} -beta ${beta} -replace_unk -verbose -gpu 3
#python translate.py -model $model -src $source -output $target.pred -beam_size 10 -alpha ${alpha} -beta ${beta} -c_attn ${c_attn} -replace_unk -verbose -gpu 3
python translate.py -model $model -src $source -output $target.pred -beam_size 10 -alpha ${alpha} -beta ${beta} -replace_unk -verbose -gpu 3
sed -r 's/(@@ )|(@@ ?$)//g' $target.pred > $target.pred.merged
perl multi-bleu.perl -lc $target < $target.pred.merged

java -Xmx2G -jar meteor-1.5/meteor-1.5.jar $target.pred.merged $target -l $tgtlang | tail -1
