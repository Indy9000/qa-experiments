from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import csv

BASE_DIR = './data'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
EMBEDDINGS_FILE = 'glove.6B.100d.txt'
WANG_DATA_DIR = BASE_DIR + '/wang/'
MAX_SEQUENCE_LENGTH = 70
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# ----------------------------------------------------------
# prepare text samples and their labels
print('Processing text dataset')

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
    nb_questions = len(qs.items())
    print('Found %s questions' % nb_questions)
    return qs

qs = load_dataset('train.csv')
nb_questions = len(qs.items())
# Split the dataset into training and validation sets on the questions

question_indices = np.arange(nb_questions)
np.random.shuffle(question_indices)
print('Shuffled questions',question_indices)
nb_val_questions = int(VALIDATION_SPLIT * nb_questions)
print('val_questions=',nb_val_questions, nb_questions,VALIDATION_SPLIT)

q_tr = question_indices[:-nb_val_questions]
q_va = question_indices[-nb_val_questions:]

print('training set=',q_tr)
print('validation set=',q_va)

qa_lines =[]
labels=[]
www = [g for g in qs.items()] #dictionary key values as a list

val_start_index = 0
# Add the training set to the big text array
for ii in q_tr:     #iterate in the order of shuffled train indices
    (qq,ans_labels) = www[ii]
    for aa,ll in ans_labels:
        qa_lines.append(qq + " " + aa)
        label_int = 0
        if ll=='1': label_int = 1;
        labels.append(label_int)
        val_start_index += 1

# Add the validation set to the big text array
for ii in q_va:     #iterate in the order of shuffled validation indices
    (aa,ans_labels) = www[ii]
    for aa,ll in ans_labels:
        qa_lines.append(qq + " " + aa)
        label_int = 0
        if ll=='1': label_int = 1;
        labels.append(label_int)

print('val index=',val_start_index)

#Tokenize
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(qa_lines)
sequences = tokenizer.texts_to_sequences(qa_lines)
print('sequences[0]=',sequences[0])


max_s_len = -1
for s in sequences:
    if max_s_len < len(s):
        max_s_len = len(s)

print("max seq len=",max_s_len)
###

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))

print('data[0]=',data[0])
print('labels[0]=',labels[0])
print('Shape of data tensor=',data.shape)
print('Shape of label tensor=',labels.shape)

x_train = data[:val_start_index]
y_train = labels[:val_start_index]
x_val = data[val_start_index:]
y_val = labels[val_start_index:]

print('Shape of training samples', x_train.shape)
print('Shape of training labels', y_train.shape)
print('Shape of validation samples', x_val.shape)
print('Shape of validation labels', y_val.shape)
## ----------------------------------------------------------
## command line arguments
#conv1d_filter_count = int(sys.argv[1])
#conv1d_filter_kernel_size = int(sys.argv[2])
#max_pooling_window_size = int(sys.argv[3])
##dense_units = 15
#dropout_rate = float(sys.argv[4])
#batch_size = int(sys.argv[5])
#epoch_count = int(sys.argv[6])
#print('filter-count = ',conv1d_filter_count,'kernel-size=',conv1d_filter_kernel_size,
#    'pooling-window=',max_pooling_window_size,'dropout=',dropout_rate,
#    'batch-size=',batch_size,'epoch-count=',epoch_count)
