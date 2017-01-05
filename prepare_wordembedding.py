# prepare word embeddings based on unique tokens in input sets
import constants
import numpy
import data_creator
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import os
import sys
import pickle
embeddings_path = 'wordembeddings/glove.6B/glove.6B.50d.txt'
# fix random seed for reproducibility
numpy.random.seed(constants.SEED)
EMBEDDING_DIM = constants.EMBEDDING_DIM

def getText(filename):
    text = ''
    with open(filename,'r') as f:
        text = f.read()
        f.close()
    return text
        
#load samples
text = getText('testextracts')
text = text + getText('extracts')
texts =  [text]
#tokenizer
tokenizer = Tokenizer(nb_words=None, lower=True, split=" ")
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#newtext = ['Good evening sir'] Stil add punctuation to word_index
#tokens = tokenizer.texts_to_sequences(newtext)
#print tokens
#sys.exit()

#save tokenizer
with open('tokenizer','wb') as ofile:
    pickle.dump(tokenizer, ofile)
    ofile.close()

#save word index to a file
with open('word_index','wb') as ofile:
    pickle.dump(word_index, ofile)
    ofile.close()

#prepare word embeddings matrix
embeddings_index = {}
f = open(os.path.abspath(os.path.join(os.pardir,embeddings_path)))
for line in f:
    values = line.split()
    word = values[0]
    coefs = numpy.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = numpy.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

#save embeddings matrix to a file
with open('emb_matrix','w') as ofile:
    numpy.savetxt(ofile, embedding_matrix)
    ofile.close()

