
#	-git clone https://github.com/facebookresearch/fastText.git fastText
#	-cd fastText
#	-make
# this doesn't work, you have to be sudo
install_fastText:
	rm -rf fastText
	git clone https://github.com/salestock/fastText.py.git fastText
	make -C fastText

DIR=/home/ubuntu/users/corp/work
data:
	ln -s $(DIR)/JRC-Acquis.en-es.es.tcs.train.bpe.shuf es.train
	ln -s $(DIR)/JRC-Acquis.en-es.es.tcs.dev.bpe.shuf es.dev
	ln -s $(DIR)/JRC-Acquis.en-es.es.tcs.test.bpe.shuf es.test
	
	ln -s $(DIR)/JRC-Acquis.en-es.en.tcs.train.bpe.shuf en.train
	ln -s $(DIR)/JRC-Acquis.en-es.en.tcs.dev.bpe.shuf en.dev
	ln -s $(DIR)/JRC-Acquis.en-es.en.tcs.test.bpe.shuf en.test

demo:
	fastText/fasttext skipgram -input es.train -output model

demo2:
	fastText/fasttext print-word-vectors model.bin < es.dev

demo3:
	fastText/fasttext supervised -input es.train -output model


