# prepare word embeddings and input representation
import constants
import numpy
import data_creator
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import sys
# fix random seed for reproducibility
numpy.random.seed(constants.SEED)
left_window = constants.LEFT_WINDOW
right_window = constants.RIGHT_WINDOW

def getCandidateEmb(length):
    t_emb = []
    for i in range(constants.EMBEDDING_DIM):
        t_emb.append(-1.0)
    t_token = []
    t_token.append(t_emb)
    c_token = []
    for i in range(length):
        c_token.append(t_token)
    c_token = numpy.array(c_token)
    return c_token
    #print c_token.shape
    
def prepare_data(filename):
    #load samples
    left, right, labels = data_creator.extractwindows(filename)

    #load tokenizer and word_index
    pkl_file = open('word_index', 'rb')
    word_index = pickle.load(pkl_file)
    pkl_file.close()
    pkl_file = open('tokenizer', 'rb')
    tokenizer = pickle.load(pkl_file)
    pkl_file.close()

    #tokenize and pad data
    left_sequences = tokenizer.texts_to_sequences(left)
    left_data = pad_sequences(left_sequences, maxlen=left_window)
    right_sequences = tokenizer.texts_to_sequences(right)
    right_data = pad_sequences(right_sequences, maxlen=right_window)

    # labels as numeric array
    labels = numpy.array(labels)
        
    #save the data and labels arrays to a file
    with open('input_rep_left_data_'+filename,'w') as ifile:
        numpy.savetxt(ifile, left_data)
        ifile.close()

    #save the data and labels arrays to a file
    with open('input_rep_right_data_'+filename,'w') as ifile:
        numpy.savetxt(ifile, right_data)
        ifile.close()
        
    with open('input_rep_label_'+filename,'w') as ifile:
        numpy.savetxt(ifile, labels)
        ifile.close()

    print('Shape of left data tensor:', left_data.shape)
    print('Shape of right data tensor:', right_data.shape)
    print('Shape of label tensor:', labels.shape)   
    print "Completed"

def main():
# display some lines
    prepare_data('extracts')
    prepare_data('testextracts')

if __name__ == "__main__":
    main()

