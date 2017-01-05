#handle period correctly in extraction and embeddings

#sample and format broadcast stream text into windows and labels
import constants
import json
from collections import OrderedDict
from pprint import pprint
from nltk.tokenize import WordPunctTokenizer
window = constants.WINDOW
half_window = window/2
variant = constants.VARIANT
def extractwindows(filename):
        left, right, labels = load(filename)
        left_texts = wordstrings(left)
        right_texts = wordstrings(right)
        return left_texts, right_texts, labels

def wordstrings(texts):
        textstrs = []
        for wordlist in texts:
                create_str = ''
                for word in wordlist:
                        create_str = create_str + word + " "
                textstrs.append(create_str)
        return textstrs
              
def collectwords(block):
        word_punct_tokenizer = WordPunctTokenizer()
        words = word_punct_tokenizer.tokenize(block)
        #words = block.split(" ")
        return words

def load(filename):
        t_words = OrderedDict()
        with open(filename, 'r') as f:
                textblock = f.read()
                subblocks = textblock.split("---------------------------")
                for i in range(len(subblocks)):
                        t_words[i] = collectwords(subblocks[i])
                f.close()
        samples = createsamples(t_words)
        with open('datasamples','w') as ofile:
                 ofile.write(json.dumps(samples))
        left = []
        right = []
        y = []
        #print samples
        for sample in samples:
                y.append(sample['label'])
                left.append(sample['left'])
                right.append(sample['right'])
        return (left, right, y)

def createsamples(d):
        data = []
        #every period but the last in each block is a non boundary
        for k, wordslist in d.items():
                indices = [] #collect indices of the periods(candidates)
                for idx, elem in enumerate(wordslist):
                        if elem=='.':
                                indices.append(idx)
                if len(indices)==0:
                        break
                if len(indices)>1:
                    for i in range(len(indices)-1):
                        data.append(createsample(d, k, indices[i], 0))
                data.append(createsample(d, k, indices[len(indices)-1], 1))
        #pprint(data)               
        return data

def createsample(d, k, i, label):
        if variant == 'equal_windows':
                return createsampleEqual(d,k,i,label)
        else:
                return createsampleUnequal(d,k,i,label)


#format: candidate ito candidate in right window, remaining left window boundary=1
#ith word in this kth block is a period/candidate
def createsampleUnequal (d,k,i,label):
    sample = {}
    sample['label']=label
 
    if label==1:
         sample['right'] = collectfirstsentence(d,k+1)
    else:
         sample['right'] = collecttillnextperiod(d, k, i)
    sample['left'] = collectleftsamplewithsize(d, k, i, window - len(sample['right']))
    return sample

def collectfirstsentence(d,knext):
        words=[]
        if knext in d:
                i=0
                while i<len(d[knext]) and d[knext][i]!='.':
                        words.append(d[knext][i])
                        i+=1
        return words

def collecttillnextperiod(d,k,i):
        words=[]
        c=i+1
        while c<len(d[k]) and d[k][c]!='.':
                words.append(d[k][c])
                c+=1
        return words

def collectleftsamplewithsize(d,k,i,length):
        words= []
        if i-length<0:
                words= words + d[k][0: i]
        else:
                words= words + d[k][i-length: i]
        prevblock=k-1
        while len(words)<length:
                if prevblock in d and len(d[prevblock])>0:
                        lastindex=len(d[prevblock])-1
                        startindex = lastindex-length+len(words)
                        if startindex<0:
                                startindex=0
                        words= d[prevblock][startindex:lastindex] + words
                        prevblock = prevblock - 1
                else:
                        break
        return words                           

#format: candidate in middle of window sampled, boundary=1
def createsampleEqual (d,k,i,label):
    sample = {}
    sample['label']=label
    sample['left'] = collectleftsample(d, k, i)
    sample['right'] = collectrightsample(d, k, i)
    return sample
   
def collectleftsample(d, k, i):
        words= []
        if i-half_window<0:
                words= words + d[k][0: i]
        else:
                words= words + d[k][i-half_window: i]
        prevblock=k-1
        while len(words)<half_window:
                if prevblock in d and len(d[prevblock])>0:
                        lastindex=len(d[prevblock])-1
                        startindex = lastindex-half_window+len(words)
                        if startindex<0:
                                startindex=0
                        words= d[prevblock][startindex:lastindex] + words
                        prevblock = prevblock - 1
                else:
                        break
        return words
        

def collectrightsample(d, k, i):
        words= []
        words= words + d[k][i+1: i+1+half_window]
        nextblock=k+1
        while len(words)<half_window:
                if nextblock in d and len(d[nextblock])>0:
                        words = words+d[nextblock][0:half_window-len(words)]
                        nextblock = nextblock + 1
                else:
                        break
        return words
