import gensim
from nltk.corpus import brown
model = gensim.models.Word2Vec(brown.sents())
model.save('brown.embedding')
brown_model = gensim.models.Word2Vec.load('brown.embedding')