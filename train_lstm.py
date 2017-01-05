# LSTM with dropout for sequence classification
#what is NB_WORDS, how is tokenizer assigning word token ids
#utilize padding, conditional distribution caputring
#capture proportion of samples from each category
import constants
import numpy
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, merge, Activation
from keras.layers.embeddings import Embedding
from keras.layers.core import Reshape, Flatten
from sklearn.model_selection import StratifiedKFold
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
skf = StratifiedKFold(n_splits=5)
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

x_train_left = neg_x_left[0:int(len(pos_y)*2)]
x_train_right = neg_x_right[0:int(len(pos_y)*2)]
y_train = neg_y[0:int(len(pos_y)*2)]

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

#create model
#trainable embeddings
shared_embedding_layer = Embedding(output_dim=EMBEDDING_DIM, input_dim= len(word_index) + 1, weights=[embedding_matrix], mask_zero=True)
shared_lstm_layer = LSTM(EMBEDDING_DIM)
left_input = Input(shape=(left_window,), dtype='float32', name='left_input')
right_input = Input(shape=(left_window,), dtype='float32', name='right_input')
lstm_left_out = shared_lstm_layer(shared_embedding_layer(left_input))
lstm_right_out = shared_lstm_layer(shared_embedding_layer(right_input))
reshape_layer = Reshape((1, EMBEDDING_DIM))
lstm_left_out = reshape_layer(lstm_left_out)
lstm_right_out = reshape_layer(lstm_right_out)
x = merge([lstm_left_out, lstm_right_out], mode='cos')
x = Reshape((1,))(x)
main_output=Activation('sigmoid', name = 'main_output')(x)
model = Model(input=[left_input, right_input], output=[main_output])
print model.summary()
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1}, metrics=['fmeasure','precision', 'recall'])
model.fit({'left_input': left_window_data, 'right_input': right_window_data},
          {'main_output': y_train},
          nb_epoch=1, batch_size=128)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
