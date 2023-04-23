import numpy as np
from underthesea import word_tokenize
from gensim.models import Word2Vec
from config import *
from utils import *

w2v_model = Word2Vec.load(W2V_MODEL)
vocab = w2v_model.wv.vocab.keys()

def embed_sentence(text):
    matrix = np.zeros((SEQUENCE_LENGTH, EMBEDDING_SIZE))
    tokens = word_tokenize(text)
    for i in range(SEQUENCE_LENGTH):
        idx = i % len(tokens)
        if tokens[idx] in vocab:
            matrix[i] = w2v_model.wv[tokens[idx]]
    matrix = np.array(matrix)
    return matrix

def CNN_predict(text, model):
    text = preprocess_sentence(text)
    x = np.expand_dims(embed_sentence(text), axis=0)
    x = np.expand_dims(x, axis=3)
    result = model.predict(x)
    return np.argmax(result)