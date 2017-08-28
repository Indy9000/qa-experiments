from __future__ import print_function

import os
import time
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras import backend as K
from keras import callbacks as CB

import sklearn.metrics as sklm
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EMBEDDING_DIM = 100
BASE_DIR = '/home/indy/data'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
EMBEDDINGS_FILE = 'glove.6B.' + str(EMBEDDING_DIM) + 'd.txt'
WANG_DATA_DIR = BASE_DIR + '/wang/'
MAX_SEQUENCE_LENGTH = 70
MAX_NB_WORDS = 39000
VALIDATION_SPLIT = 0.2

ttt0 = time.time()
# ----------------------------------------------------------
# prepare text samples and their labels
print('Processing text dataset')

# Loads the dataset into a dictionary where key is the question
# Values are the list of tuples. Each tuple is an answer and the label
def load_dataset(filename):
    qs = {}
    nb_questions = 0
    with open(WANG_DATA_DIR + filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',',quotechar='"')
        next(reader,None) #skip header (question,label,answer)
        for row in reader:
            question = row[0]; label = row[1]; answer = row[2]
            if question in qs:
                qs[question].append((answer,label))
            else:
                qs[question]= [(answer,label)]
    print('Found %s questions' % nb_questions)
    return qs

# Load the dataset
dataset_file = 'train-all.csv'
testset_file = 'test.csv'
qs = load_dataset(dataset_file)
qs_test = load_dataset(testset_file)

nb_questions = len(qs.items())
nb_test_questions = len(qs_test.items())

all_q = [g for g in qs.items()] #dictionary key values as a list
test_q = [q for q in qs_test.items()]

print("Dataset ",dataset_file," loaded. question count=",len(all_q))
print("Testset ",testset_file," loaded. question count=",len(test_q))

cc_total = 0
anc_total = 0
cc_list=[]
anc_list=[]
for q in qs.keys():
    cc = 0 #correct count
    anc= len(qs[q]) #total answer count per question
    for (a,l) in qs[q]:
        if l=='1':
            cc += 1
            cc_total += 1
    #print("CC/Ans=",cc,anc)
    cc_list.append(cc)
    anc_list.append(anc)
    anc_total += anc

print("Correct answers=",cc_total,"total Answers=",anc_total, float(cc_total)/float(anc_total),"%")
print("TRAIN cc-histo=",np.histogram(cc_list,bins=[0,2,5,8,10,15,20,25,30,35,40]))
print("TRAIN anc-histo=",np.histogram(anc_list,bins=[0,25,50,75,100,200,500,700]))

plt.hist(cc_list, bins=[0,2,5,8,10,15,20,25,30,35,40])  # arguments are passed to np.histogram
plt.title("Histgram of correct answer count per question - TRAIN-ALL")
plt.savefig("train-all-cc.png")
plt.close()


plt.hist(anc_list, bins=[0,25,50,75,100,200,500,700])  # arguments are passed to np.histogram
plt.title("Histogram of answer count per question - TRAIN-ALL")
plt.savefig("train-all-anc.png")
plt.close()
#######################################################################################
cc_list=[]
anc_list=[]
cc_total = 0
anc_total = 0
for q in qs_test.keys():
    cc = 0 #correct count
    anc= len(qs_test[q]) #total answer count per question
    for (a,l) in qs_test[q]:
        if l=='1':
            cc += 1
            cc_total += 1
    #print("CC/Ans=",cc,anc)
    cc_list.append(cc)
    anc_list.append(anc)
    anc_total += anc

print("Correct answers=",cc_total,"total Answers=",anc_total, float(cc_total)/float(anc_total),"%")
print("TEST max-cc",max(cc_list), "min-cc",min(cc_list))
print("TEST max-anc",max(anc_list), "min_anc",min(anc_list))

print("TEST cc-histo=",np.histogram(cc_list,bins=[0,2,5,8,10,15]))
print("TEST anc-histo=",np.histogram(anc_list,bins=[0,10,20,30,40,50,75,100]))

plt.hist(cc_list, bins=[0,2,5,8,10,15])  # arguments are passed to np.histogram
plt.title("Histogram of correct answer count per question - TEST")
plt.savefig("test-all-cc.png")
plt.close()


plt.hist(anc_list, bins=[0,10,20,30,40,50,75,100])  # arguments are passed to np.histogram
plt.title("Histogram of answer count per question - TEST")
plt.savefig("test-all-anc.png")
plt.close()
exit()
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Shuffle question indices
question_indices = np.arange(nb_questions)
test_question_indices = np.arange(nb_test_questions)

# Training and validation set should be shuffled by question indices
np.random.shuffle(question_indices)

#print('Shuffled questions',question_indices)
nb_va_questions = int(VALIDATION_SPLIT * nb_questions)
print('va_questions=',nb_va_questions,"out of", nb_questions,"with a split of ",VALIDATION_SPLIT)

# Split the dataset into training and validation sets on the questions
qi_tr = question_indices[:-nb_va_questions]
qi_va = question_indices[-nb_va_questions:]
qi_te = test_question_indices

print('training set=', len(qi_tr))
print('validation set=', len(qi_va))
print('test set=',len(qi_te))

print('\n qi_tr=', qi_tr)
print('\n qi_va=', qi_va)
print('\n qi_te=',qi_te)

class QuestionAnswerSet:
    def __init__(self,qi):
        self.QuestionIndex = qi
        self.CorrectAnswers = [] # There can be zero or more correct answers
        self.AnswerCount   = 0
        self.PredictedList = []

# Splice the question and answer into a single line of all training and validation sets
# before sending to the tokeniser and create vocabulary.

def vectorise_qa_lines_labels(qi,qs,qa_lines,labels,qas_list,qii):
    k=0 # represent the unshuffled question index
    for ii in qi:     # iterate in the order of question indices
        qas = QuestionAnswerSet(ii)
        (qq,ans_labels) = qs[ii] # get the q and the list of tuples(ans,label)
        a_counter = 0
        for aa,ll in ans_labels:    # for each tuple expand and add to lines
            qa_lines.append(qq + " " + aa)
            qii.append(k) #this stores the question index of the qas for each qa_line
            # decodes the label 
            label_int = 0
            if ll=='1': 
                label_int = 1;
                qas.CorrectAnswers.append(a_counter)
            # count the number of answers for this question
            a_counter += 1
            labels.append(label_int)
        k += 1 #increment for each question
        qas.AnswerCount = a_counter
        qas_list.append(qas)
    cc=0
    for qz in qas_list:
        cc += qz.AnswerCount
    print("Total answercount=",cc)

full_qa_lines=[]
full_labels  =[]

tr_questions=[]# This list will hold objects of QuestionAnswerSet so that we can compute MAP after the model had been trained
tr_qii=[]
va_questions=[]# This list will hold objects of QuestionAnswerSet so that we can compute MAP after the model had been trained
va_qii=[]
te_questions=[]# This list will hold objects of QuestionAnswerSet so that we can compute MAP after the model had been trained
te_qii=[]
va_start_index = 0 #since we are having both training and val set in the same list, we
                    #need to keep track of where the val set starts

# First add the training set to the big text array
vectorise_qa_lines_labels(qi_tr, all_q, full_qa_lines, full_labels, tr_questions, tr_qii)
va_start_index = len(full_qa_lines)
vectorise_qa_lines_labels(qi_va, all_q, full_qa_lines, full_labels, va_questions, va_qii)
te_start_index = len(full_qa_lines)
# Then add the test set to the big text array
vectorise_qa_lines_labels(qi_te, test_q, full_qa_lines, full_labels, te_questions, te_qii)
print('len(full_qa_lines)=', len(full_qa_lines), 'len(full_labels)=',len(full_labels))
print('val start index=', va_start_index, "test start index=", te_start_index)
#print("va_questions=", va_questions[0].QuestionIndex, va_questions[0].CorrectAnswers, va_questions[0].AnswerCount, va_questions[0].PredictedList)


# Tokenize the text lines to words and filter and clean up
tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(full_qa_lines)
sequences = tokenizer.texts_to_sequences(full_qa_lines)
print('sequences[0]=',sequences[0])
print('len(sequences)=',len(sequences))
# Find the max sequence length
max_s_len = -1
for s in sequences:
    if max_s_len < len(s):
        max_s_len = len(s)

print("max seq len=",max_s_len)

vocabulary = tokenizer.word_index
print('Found %s unique tokens' % len(vocabulary))

full_dataset = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
full_labels  = np.asarray(full_labels)

print('full_dataset[0]=',full_dataset[0])
print('full_labels[0]=',full_labels[0])

print('Shape of full_dataset tensor=',full_dataset.shape)
print('Shape of full_label tensor=',full_labels.shape)
print('len(full_labels)',len(full_labels))

x_tr = full_dataset[:va_start_index]
y_tr = full_labels[:va_start_index]
x_va = full_dataset[va_start_index: te_start_index]
y_va = full_labels[va_start_index: te_start_index]
x_te = full_dataset[te_start_index:]
y_te = full_labels[te_start_index:]

print('Shape of training samples', x_tr.shape)
print('Shape of training labels', y_tr.shape)
print('Shape of validation samples', x_va.shape)
print('Shape of validation labels', y_va.shape)
print('Shape of test samples', x_te.shape)
print('Shape of test labels', y_te.shape)

print('Shape of full_dataset[te_start_index+10]',full_dataset[te_start_index+10])

###################################################################
# Save vocabulary
#with open(WANG_DATA_DIR + 'vocabulary.txt', 'w') as voc:
#   for word,index in vocabulary.items():
#       voc.write(word+'\n')
###################################################################

#-----------------------------------------------------------------
# first, build index mapping words in the embeddings set
# to their embedding vector

print('Loading word vectors.')
embeddings_index = {}
with open(os.path.join(GLOVE_DIR,EMBEDDINGS_FILE),'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# create the embedding matrix
words_without_embeddings = 0
# Initialise embedding matrix with zeros
embedding_matrix = np.zeros((len(vocabulary) + 1, EMBEDDING_DIM))
for word, i in vocabulary.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is None:
        # words not found in embedding index will be all zeros
        #print('Word Not found in embeddings',word)
        words_without_embeddings += 1
    else:
        embedding_matrix[i] = embedding_vector
        #print('embedding vector = ', embedding_vector)

print('words without embeddings =', words_without_embeddings)

