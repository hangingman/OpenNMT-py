import fasttext
import numpy as np
import pickle



class VectorSearchEngine:
	MODEL_PATH = "./model"
	SENT_VEC = "./sent_vectors.pic"

	def create_model(self, file):
		"""file: tokenized training file, one sentence per line"""
		skipgram = fasttext.skipgram(
			file, self.MODEL_PATH, lr=0.02, dim=300, ws=5,
			epoch=1, min_count=5, neg=5, loss='ns', bucket=2000000, minn=3, maxn=6,
			thread=4, t=1e-4, lr_update_rate=100
			)

		self.skipgram = skipgram
		self.save_sentences_as_vectors(file)

	def load_model(self):
		skipgram = fasttext.load_model(self.MODEL_PATH + ".bin")
		self.skipgram = skipgram
		self.sent_vectors = pickle.load(open(self.SENT_VEC, "rb"))

	def save_sentences_as_vectors(self, file):
		self.sent_vectors = []
		for sent in open(file):
			sent_v = self.get_sent_vec(sent.split())
			self.sent_vectors.append(sent_v)
		self.sent_vectors = np.array(self.sent_vectors)
		pickle.dump(self.sent_vectors, open(self.SENT_VEC, "wb"))


	def get_word_vec(self, word):
		"""word: string"""
		v = np.array(self.skipgram[word])
		# normalized to sum to 1
		v /= sum(v)
		return v

	def get_sent_vec(self, sentence):
		"""sentence: list of strings"""
		wv = sum(self.get_word_vec(w) for w in sentence) / len(sentence)
		return wv


	def search(self, sentence, limit1=5, limit2=5):
		"""sentence: list of strings"""
		v = self.get_sent_vec(sentence)
#		for s in 
		



if __name__ == "__main__":
	import sys
	se = VectorSearchEngine()
	#se.create_model(sys.argv[1])
	se.load_model()
	
	t = ["what", "the" ]
	se.search(t)
