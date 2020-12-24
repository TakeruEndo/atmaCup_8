import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm



# this function creates a normalized vector for the whole sentence
def sent2vec(s):
    words = str(s).lower()
    words = nltk.word_tokenize(words)
    # words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())


def glove_embedding(df, target):
    # load the GloVe vectors in a dictionary:
    embeddings_index = {}
    f = open('../../models/glove.840B.300d.txt')
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except:
            embeddings_index[word] = 0
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = nltk.corpus.stopwords.words('english')

    # create sentence vectors using the above function for training and validation set
    xtrain_glove = [sent2vec(x) for x in tqdm(df[target].values)]
    xtrain_glove = np.array(xtrain_glove)
