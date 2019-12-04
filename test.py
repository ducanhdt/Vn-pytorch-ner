from gensim.models import KeyedVectors as FastText

model = FastText.load_word2vec_format("./data/VNdata/cc.vi.300.vec")

print(model['hay'])
