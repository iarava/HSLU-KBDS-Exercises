import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

IMDB_1 = open('IMDB_1.txt', 'r').read()
IMDB_2 = open('IMDB_2.txt', 'r').read()
IMDB_3 = open('IMDB_3.txt', 'r').read()

documents = [
    IMDB_1,
    IMDB_2,
    IMDB_3
]

document_names = ['Doc {:d}'.format(i) for i in range(len(documents))]

def get_tfidf(docs, ngram_range=(1,1), index=None):
    vect = TfidfVectorizer(ngram_range=ngram_range)
    tfidf = vect.fit_transform(documents).todense()
    return pd.DataFrame(tfidf, columns=vect.get_feature_names(), index=index).T

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(get_tfidf(documents, ngram_range=(1,1), index=document_names))