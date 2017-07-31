from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from keras.utils import to_categorical
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

# command line arguments
conv1d_filter_count = int(sys.argv[1])
conv1d_filter_kernel_size = int(sys.argv[2])
max_pooling_window_size = int(sys.argv[3])
#dense_units = 15
dropout_rate = float(sys.argv[4])
batch_size = int(sys.argv[5])
epoch_count = int(sys.argv[6])
print('filter-count = ',conv1d_filter_count,'kernel-size=',conv1d_filter_kernel_size,
        'pooling-window=',max_pooling_window_size,'dropout=',dropout_rate,
        'batch-size=',batch_size,'epoch-count=',epoch_count)
# ----------------------

# prepare text samples and their labels
print('Processing text dataset')

texts = [] # list of text samples
labels = [] # list of label ids
labels_index = {} # dictionary mapping label name to numeric id

def load_dataset(filename):
    with open(WANG_DATA_DIR + filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader, None) #skip header (question,label,answer)
        count = 0
        for row in reader:
            qa = row[0]+" "+ row[2] #append question and answer
            label = row[1]
            texts.append(qa)
            label_id = 0
            if label in labels_index:
                label_id = labels_index[label]
            else:
                label_id = len(labels_index)
                labels_index[label] = label_id 
            labels.append(label_id)
            count +=1
            if count == 10:
                break

    print('Found %s samples' % len(texts))

load_dataset('train.csv')

# tokenize 
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

###
max_s_len = -1
for s in sequences:
    if max_s_len < len(s):
        max_s_len = len(s)

print("max seq len=",max_s_len)
###

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

#labels = to_categorical(np.asarray(labels))
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

print('Before shuffled Labels::::: ', labels)
# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
print('Shuffled indices:::: ',indices)
data = data[indices]
labels = labels[indices]
print('After shuffled Labels::::: ', labels)

nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
print('Shape of training samples', x_train.shape)
print('Shape of training labels', y_train.shape)
print('Shape of validation samples', x_val.shape)
print('Shape of validation labels', y_val.shape)

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
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is None:
        # words not found in embedding index will be all zeros
        #print('Word Not found in embeddings',word)
        words_without_embeddings += 1
    else:
        embedding_matrix[i] = embedding_vector
        #print('embedding vector = ', embedding_vector)

#print('words without embeddings =', words_without_embeddings)

# load the embedding matrix into a frozen layer
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

#-------------------------------------------------------------------

#conv1d_filter_count = sys.argv[1]
#conv1d_filter_kernel_size = sys.argv[2]
#max_pooling_window_size = sys.argv[3]
#dropout_rate = sys.argv[4]
#batch_size = sys.argv[5]
#epoch_count = sys.argv[6]

# build model
print('len(labels)',len(labels))

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
print('shape inp = ', sequence_input.shape)
embedded_sequences = embedding_layer(sequence_input)
print('shape embed = ', embedded_sequences.shape)
x = Conv1D(conv1d_filter_count,conv1d_filter_kernel_size, activation='relu')(embedded_sequences)
print('shape conv1d = ',x.shape)
x = MaxPooling1D(max_pooling_window_size)(x)
print('shape maxp1d = ',x.shape)
x = Flatten()(x)
print('shape flat = ',x.shape)
#x = Dense(15, activation='relu')(x)
#print('shape dense = ',x.shape)
x = Dropout(dropout_rate)(x)
#x = Dense(5, activation='relu')(x)
#print('shape dense = ',x.shape)

#preds = Dense(len(labels_index), activation='softmax')(x)
preds = Dense(1, activation='softmax')(x)
print('shape dense = ',preds.shape)
model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy', optimizer='adam', 
                metrics=['binary_accuracy'])

#-------------------------------------------------------------------
# start learning
model.fit(x_train, y_train, validation_data=(x_val, y_val),
            epochs=epoch_count, batch_size=batch_size)
	
