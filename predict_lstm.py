import constants
import prepare_data
import numpy
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, LSTM, Input, merge, Activation
from keras.layers.embeddings import Embedding
from keras.layers.core import Reshape
from pprint import pprint
import pickle
import sys

# fix random seed for reproducibility
numpy.random.seed(constants.SEED)
left_window = constants.LEFT_WINDOW
right_window = constants.RIGHT_WINDOW
EMBEDDING_DIM = constants.EMBEDDING_DIM

#load prediction/test data
testleft = numpy.loadtxt('input_rep_left_data_testextracts')
testright = numpy.loadtxt('input_rep_right_data_testextracts')
testlabel = numpy.loadtxt('input_rep_label_testextracts')

#numpy vectors for training data
left_window_data = numpy.asmatrix(testleft)
right_window_data = numpy.asmatrix(testright)
labels = numpy.asarray(testlabel)

#load word_index and embedding_matrix
embedding_matrix = numpy.loadtxt('emb_matrix')
pkl_file = open('word_index', 'rb')
word_index = pickle.load(pkl_file)
pkl_file.close()

#load model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")

print "Loading Completed.."

print model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['fmeasure','precision', 'recall'])

#c_token = prepare_data.getCandidateEmb(len(left_window_data))


# Final evaluation of the model
scores = model.evaluate([left_window_data,right_window_data], labels, verbose=0)
print("Fmeasure: %.2f%%" % (scores[1]*100))
print("Precision: %.2f%%" % (scores[2]*100))
print("Recall: %.2f%%" % (scores[3]*100))

# calculate predictions
predictions = model.predict([left_window_data,right_window_data])

#save predictions
with open('predictions','wb') as ofile:
    pickle.dump(predictions, ofile)
    ofile.close()
'''
pkl_file = open('predictions', 'rb')
predictions = pickle.load(pkl_file)
pkl_file.close()
'''
'''
#dp
#predictions[0] = for each candidate prob of boundary given prev label is b
#predictions[1] = for each candidate prob of boundary given prev label is nb

dp=[[0]*len(predictions) for _ in range(2)]
#dp[0] prob of seq ending in b
#dp[1] prob of seq ending in nb
dp[0][0] = predictions[0][0]
dp[1][0] = 1-predictions[0][0]
or i in range(1,len(predictions)):
    #boundary
    dp[0][i] = max(dp[0][i-1]*predictions[0][i] + dp[1][i-1]*predictions[1][i])
    #nb
    dp[1][i] = max(dp[0][i-1]*predictions[0][i] + dp[1][i-1]*predictions[1][i])

for i in range(len(dp)):
    for j in range(len(dp[0])):
        print dp[i][j]
    print "\n"
   
#round predictions
predictions = numpy.reshape(predictions, (1,len(predictions)))
predictions = predictions.tolist()
predictions = predictions[0]
''' 
#print predictions
rounded = [1 if x> 0.49 else 0  for x in predictions]
#print(map(int,rounded))
#print(map(int, labels))

# false alarm: was declared if the segmentation algorithm placed a boundary in the interval while no reference boundary existed in the interval,
# miss: was declard if the segmentation didn't place a boundary in the interval while a reference boundary did exist in the interval
    
with open('testextracts', 'r') as f:
    false_alarm_count=0
    miss_count=0
    hit_count = 0
    stream = f.read()
    stream = stream.replace("---------------------------", "#")
    stream = stream.strip(" ")
    index = 0
    wordcount = 0
    bflag = False
    indices = []
    windowcounts = 0
    word = []
    for charindex in range(len(stream)):
        car = stream[charindex]
        word.append(car)
        if(car==' '):
            wordcount+=1
            #print word
            #print wordcount
            word= []
        elif(car == '#'):
            bflag=True
        elif(car=='.'):
            indices.append(index)
            index = index+1
        if wordcount==50:
            windowcounts+=1
            count1 = 0
            #print indices
            for index in indices:
                if rounded[index]==1:
                    count1+=1
            #print count1, bflag
            if(count1>1):
                print "Warning: Multiple boundaries in 50 word window: " + str(count1)
            if(count1==0 and bflag==1):
                miss_count += 1
            elif(count1>0 and bflag==0):
                false_alarm_count += 1
            elif(count1>0 and bflag==1):
                hit_count += 1
            #print count1
            #reset
            wordcount=0
            bflag = False
            indices = []
    print "Total predictions:" + str(len(rounded))
    print "Total windows:"+ str(windowcounts)
    print "fa:"+str(false_alarm_count)
    print "miss:"+str(miss_count)
    print "hit_count:"+str(hit_count)
    precision = hit_count*1.0/(hit_count+false_alarm_count)
    recall = hit_count*1.0/(hit_count+miss_count)
    f1 = 2* precision * recall/(precision+recall)
    print "F1: %f",f1
    print "Precision: %f", precision
    print "Recall: %f", recall

           
'''
--------------------------------------------------------
#tolerance
Recall is the fraction of reference boundaries that are retrieved.
Precision is the fraction of de- clared boundaries that coincide with
reference boundaries. 

for every boundary annotation, check if for candidates within 50 words, a boundary was marked

def pick50backwards(s):
    #starting at segment s
    words = s.split(" ")
    for w in reversed(words):
        
with open('testextracts', 'r') as f:

    seg_indices = []
    stream = f.read()
    segments = stream.split("---------------------------")
    global_sent = []
    wordspersentence = []
    seg_cand = -1
    for seg in segments:
        if len(seg)>1:
            sentences = seg.split(".")
            seg_cand = seg_cand + len(sentences)
            seg_indices.append(seg_cand)
            global_sent = global_sent + sentences
    for sent in global_sent:
            sent = sent.strip(" ")
            words = sent.split(" ")
            if (len(words)==1) and words[0] == '\n':
                wordspersentence.append(-1)
            else:
                wordspersentence.append(len(words))
    pprint(global_sent)
    pprint(seg_indices)
    pprint(wordspersentence)
    #indices of period within 50 words of splitting point (50 to its left and 50 right)
    seg_ind = {}
    for s in seg_indices:
        w=0
        c=s+1
        seg_range = []
        while c<len(wordspersentence) and w+wordspersentence[c]<=50:
            if wordspersentence[c]!=-1:
                   w = w + wordspersentence[c]
                   seg_range.append(c)
            c=c+1
        seg_ind[s] = seg_range
    pprint(seg_ind)
'''          
'''two shared lstms + cos
Total predictions:699
Total windows:215
fa:84
miss:12
hit_count:40
F1: %f 0.454545454545
Precision: %f 0.322580645161
Recall: %f 0.769230769231
'''
'''cnn with c_token
Total predictions:699
Total windows:215
fa:103
miss:20
hit_count:32
F1: %f 0.342245989305
Precision: %f 0.237037037037
Recall: %f 0.615384615385
'''
