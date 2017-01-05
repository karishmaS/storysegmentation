# LSTM with dropout for sequence classification
#what is NB_WORDS, how is tokenizer assigning word token ids
#utilize padding, conditional distribution caputring
#capture proportion of samples from each category
from keras.layers import Input, merge, Dense
from keras.models import Model
import numpy as np

#load training and validation data and labels
left = numpy.loadtxt('input_rep_left_data')
right = numpy.loadtxt('input_rep_right_data')
labels = numpy.loadtxt('input_rep_label')

#split into training and validation for skewed classes
skf = StratifiedKFold(n_splits=2)
for train_index, test_index in skf.split(left, labels):
    print x_train_left
    print y_train
    x_train_left=left[train_index]
    x_train_right=right[train_index]
    x_val_left=left[test_index]
    x_val_right = right[test_index]
    y_train, y_val = labels[train_index], labels[test_index]

'''
input_a = np.reshape([1, 2, 3, 5, 6], (1, 1, 5))
input_b = np.reshape([409, 520, 6, 1, 5], (1, 1, 5))

a = Input(shape=(5))
b = Input(shape=(5))

concat = merge([a, b], mode='concat', concat_axis=-1)
dot = merge([a, b], mode='dot', dot_axes=2)
cos = merge([a, b], mode='cos')
a = Reshape((1, EMBEDDING_DIM)))(a)
b = Reshape((1, EMBEDDING_DIM)))(b)
cos = Dense(1,activation = 'sigmoid')(cos)
mul = merge([a, b], mode='mul', dot_axes=2)

model_concat = Model(input=[a, b], output=concat)
model_dot = Model(input=[a, b], output=dot)
model_cos = Model(input=[a, b], output=cos)
model_mul = Model(input = [a,b], output = mul)

print(model_concat.predict([input_a, input_b]))
print(model_dot.predict([input_a, input_b]))
print(model_cos.predict([input_a, input_b]))
print(model_mul.predict([input_a,input_b]))
'''
