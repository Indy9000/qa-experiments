from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import csv

BASE_DIR = './data'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
WANG_DATA_DIR = BASE_DIR + '/wang/'
MAX_SEQUENCE_LENGTH = 500
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


# prepare text samples and their labels
print('Processing text dataset')

texts = [] # list of text samples
labels = [] # list of label ids
labels_index = {} # dictionary mapping label name to numeric id

def load_dataset(filename):
    with open(WANG_DATA_DIR + filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader, None) #skip header (question,label,answer)
        for row in reader:
            qa = row[0] + row[2] #append question and answer
            label = row[1]
            texts.append(qa)
            label_id = 0
            if label in labels_index:
                label_id = labels_index[label]
            else:
                label_id = len(labels_index)
                labels_index[label] = label_id 
            labels.append(label_id)

    print('Found %s samples' % len(texts))

load_dataset('train.csv')

# tokenize 
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
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
f = open(os.path.join(GLOVE_DIR,'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# create the embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all zeros
        embedding_matrix[i] = embedding_vector

# load the embedding matrix into a frozen layer
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

#-------------------------------------------------------------------
# build model
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(10,5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
# x = Conv1D(128,5, activation='relu')(x)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128,5, activation='relu')(x)
# x = MaxPooling1D(10)(x) # global max pooling
x = Flatten()(x)
x = Dense(10, activation='relu')(x)

print('len(labels)',len(labels))

preds = Dense(len(labels_index), activation='softmax')(x)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
                metrics=['acc'])

#-------------------------------------------------------------------
# start learning
model.fit(x_train, y_train, validation_data=(x_val, y_val),
            epochs=20, batch_size=10)
	
