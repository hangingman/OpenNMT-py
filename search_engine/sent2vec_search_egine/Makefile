DATA=dev.en train.en test.en dev.fr train.fr test.fr

data $(DATA):
	rm -rf $(DATA)
	ln -s /home/ubuntu/users/corp_v2/work/JRC-Acquis.en-fr.en.tcs.dev.bpe.shuf dev.en
	ln -s /home/ubuntu/users/corp_v2/work/JRC-Acquis.en-fr.en.tcs.train.bpe.shuf train.en
	ln -s /home/ubuntu/users/corp_v2/work/JRC-Acquis.en-fr.en.tcs.test.bpe.shuf test.en
	:
	ln -s /home/ubuntu/users/corp_v2/work/JRC-Acquis.en-fr.fr.tcs.dev.shuf dev.fr
	ln -s /home/ubuntu/users/corp_v2/work/JRC-Acquis.en-fr.fr.tcs.train.shuf train.fr
	ln -s /home/ubuntu/users/corp_v2/work/JRC-Acquis.en-fr.fr.tcs.test.shuf test.fr

install sent2vec:
	git clone https://github.com/Gldkslfmsd/sent2vec.git
	make -j -C sent2vec

SE=sent2vec/fasttext

demo-fr: sent2vec $(DATA)
	$(SE) nnSent /home/ubuntu/vec_models/wiki.fr.bin train.fr < dev.fr
	
demo-en:
	sent2vec/fasttext nnSent ~/vec_models/wiki.en.bin /home/ubuntu/users/corp_v2/work/JRC-Acquis.en-fr.en.dev.shuf 

demo-en2:
	sent2vec/fasttext nnSent ~/vec_models/wiki.en.bin /home/ubuntu/users/corp_v2/work/JRC-Acquis.en-fr.en.dev.shuf  < dev.en


compile:
	make -C sent2vec

demo-cs:
	sent2vec/fasttext nnSent ~/vec_models/wiki.cs.bin pejsekakocicka.txt

clean:
	rm -rf $(DATA)

demo-run:
	./run_search_eng.sh ~/vec_models/wiki.cs.bin pejsekakocicka.txt 5 rev.pejsek.txt < pejsekakocicka.txt
