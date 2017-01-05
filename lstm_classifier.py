# LSTM with dropout for sequence classification
#what is NB_WORDS,py how is tokenizer assigning word token ids
#utilize padding, conditional distribution caputring
#capture proportion of samples from each category
import constants
import data_creator
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import sys
import pickle

# fix random seed for reproducibility
numpy.random.seed(constants.SEED)
window = constants.WINDOW
EMBEDDING_DIM = constants.EMBEDDING_DIM

#load word_index and embedding_matrix
embedding_matrix = numpy.loadtxt('emb_matrix')
pkl_file = open('word_index', 'rb')
word_index = pickle.load(pkl_file)
pkl_file.close()

#prepare data and labels
#load samples
texts, labels = data_creator.extractwindows('extracts_eval')

data = []
#tokenize and pad data
for sample in texts:
    sequence = []
    sample = sample.strip()
    words = sample.split(" ")
    if(len(words)>window):
        print sample+"-"
        print words
        sys.exit("Error")
    pad = window - len(words)
    while pad>0:
        sequence.append(0)
        pad -=1
    for word in words:
        if word in word_index:
            sequence.append(word_index[word])
        else:
            sequence.append(0)
    data.append(sequence)

data = numpy.matrix(numpy.array(data))

with open('savefiled','w') as ifile:
    numpy.savetxt(ifile, data)
    ifile.close()

# labels as numeric array
labels = numpy.array(labels)

print data.shape
print labels.shape
print "Evaluation Loading Completed.."

#load model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['fmeasure', 'precision', 'recall'])
'''
print model.metrics_names
# Evaluation of the model on test set
scores = model.evaluate(data, labels, verbose=0)
print scores
'''
# calculate predictions
predictions = model.predict(data)
# round predictions
rounded = [round(x) for x in predictions]
print(rounded)
'''
#http://stackoverflow.com/questions/36700790/keras-text-classification-lstm-how-to-input-text?rq=1
http://deeplearning.net/tutorial/rnnslu.html
'''
