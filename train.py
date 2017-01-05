# LSTM with dropout for sequence classification
#what is NB_WORDS, how is tokenizer assigning word token ids
#utilize padding, conditional distribution caputring
#capture proportion of samples from each category
import constants
import prepare_data
import numpy
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, merge, Activation
from keras.layers.embeddings import Embedding
from keras.layers.core import Reshape, Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.model_selection import StratifiedKFold
from keras.models import model_from_json
import pickle
import sys

# fix random seed for reproducibility
numpy.random.seed(constants.SEED)
left_window = constants.LEFT_WINDOW
right_window = constants.RIGHT_WINDOW
EMBEDDING_DIM = constants.EMBEDDING_DIM

x_train_left = numpy.loadtxt('input_rep_left_data_extracts')
x_train_right = numpy.loadtxt('input_rep_right_data_extracts')
y_train = numpy.loadtxt('input_rep_label_extracts')

'''
#load training and validation data and labels
left = numpy.loadtxt('input_rep_left_data_extracts')
right = numpy.loadtxt('input_rep_right_data_extracts')
labels = numpy.loadtxt('input_rep_label_extracts')


#split into training and validation for skewed classes
skf = StratifiedKFold(n_splits=4)
for train_index, test_index in skf.split(left, labels):
    x_train_left=left[train_index]
    x_train_right=right[train_index]
    x_val_left=left[test_index]
    x_val_right = right[test_index]
    y_train, y_val = labels[train_index], labels[test_index]

#shuffling the training set
indices = numpy.arange(x_train_left.shape[0])
numpy.random.shuffle(indices)
x_train_left = x_train_left[indices]
x_train_right = x_train_right[indices]
y_train = y_train[indices]
'''
#equalize training samples by label/class
pos_y = []
neg_y = []
pos_x_left=[]
pos_x_right=[]
neg_x_left=[]
neg_x_right=[]
for ilabel in range(len(y_train)):
    if y_train[ilabel]==1:
        pos_y.append(y_train[ilabel])
        pos_x_left.append(x_train_left[ilabel])
        pos_x_right.append(x_train_right[ilabel])
    else:
        neg_y.append(y_train[ilabel])
        neg_x_left.append(x_train_left[ilabel])
        neg_x_right.append(x_train_right[ilabel])

x_train_left = neg_x_left[0:int(len(pos_y)*1.5)]
x_train_right = neg_x_right[0:int(len(pos_y)*1.5)]
y_train = neg_y[0:int(len(pos_y)*1.5)]

for i in range(len(pos_y)):
    x_train_left.append(pos_x_left[i])
    x_train_right.append(pos_x_right[i])
    y_train.append(pos_y[i])

#numpy vectors for training data
left_window_data = numpy.asmatrix(x_train_left)
right_window_data = numpy.asmatrix(x_train_right)
y_train = numpy.asarray(y_train)

#load word_index and embedding_matrix
embedding_matrix = numpy.loadtxt('emb_matrix')
pkl_file = open('word_index', 'rb')
word_index = pickle.load(pkl_file)
pkl_file.close()

print "Loading Completed.."

c_token = prepare_data.getCandidateEmb(len(left_window_data))

#def train_save_model():

#model 
l_inp = Input(shape=(left_window,), dtype='float32', name = 'l_inp')
r_inp = Input(shape=(right_window,), dtype='float32', name = 'r_inp')
c_emb = Input(shape=(1,50), dtype = 'float32', name='c_inp')
#mask to be added
l_emb = Embedding(output_dim = EMBEDDING_DIM, input_dim = len(word_index) + 1, weights=[embedding_matrix], mask_zero=False)(l_inp)
r_emb = Embedding(output_dim = EMBEDDING_DIM, input_dim = len(word_index) + 1, weights=[embedding_matrix], mask_zero=False)(r_inp)
merge= merge([l_emb, c_emb, r_emb], mode = 'concat', concat_axis=1)
conv1D = Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu')(merge)
maxPool = MaxPooling1D(pool_length=2)(conv1D)
d = Dense(1, activation = 'relu')(maxPool)
d = Reshape(((left_window+right_window)/2,))(d)
out = Dense(1, activation = 'sigmoid', name = 'out')(d)
model = Model(input=[l_inp, r_inp, c_emb], output=[out])
print model.summary()
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['fmeasure','precision', 'recall'])
model.fit({'l_inp':left_window_data, 'r_inp':right_window_data, 'c_inp': c_token}, {'out': y_train}, nb_epoch = 15, batch_size = 32)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

'''
def load_model():
    #load model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['fmeasure','precision', 'recall'])
    return loaded_model

loaded_model = load_model()

# evaluate loaded model on test data
c_token = prepare_data.getCandidateEmb(len(x_val_left))
scores = loaded_model.evaluate([x_val_left, x_val_right, c_token], y_val, verbose=0)
print("Fmeasure: %.2f%%" % (scores[1]*100))
print("Precision: %.2f%%" % (scores[2]*100))
print("Recall: %.2f%%" % (scores[3]*100))

Fmeasure: 22.21%
Precision: 16.80%
Recall: 52.25%
'''
'''
Total predictions:2599
Total windows:852
fa:453
miss:36
hit_count:168
F1: %f 0.407272727273
Precision: %f 0.270531400966
Recall: %f 0.823529411765
'''



