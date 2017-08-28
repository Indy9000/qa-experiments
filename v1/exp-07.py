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
from keras import backend as K
from keras import callbacks as CB

import sklearn.metrics as sklm
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

qs = load_dataset('train-all.csv')
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


class ValQuestion:
    def __init__(self,qi):
        self.QuestionIndex = qi
        self.CorrectAnswer = -1
        self.AnswerCount   = 0
        self.PredictedList = []

# This list will hold objects of ValQuestions so that we can compute MAP 
# After the model had been trained

validation_questions=[]

# Add the validation set to the big text array
for ii in q_va:     #iterate in the order of shuffled validation indices
    (aa,ans_labels) = www[ii]
    vq = ValQuestion(ii)
    a_counter = 0
    for aa,ll in ans_labels:
        qa_lines.append(qq + " " + aa)
        label_int = 0
        if ll=='1': 
            label_int = 1;
            vq.CorrectAnswer = a_counter
        a_counter += 1
        labels.append(label_int)
    vq.AnswerCount = a_counter
    validation_questions.append(vq)

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


word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.asarray(labels)

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

###################################################################
# Save vocabulary
#with open(WANG_DATA_DIR + 'vocabulary.txt', 'w') as voc:
#   for word,index in word_index.items():
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
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is None:
        # words not found in embedding index will be all zeros
        print('Word Not found in embeddings',word)
        words_without_embeddings += 1
    else:
        embedding_matrix[i] = embedding_vector
        #print('embedding vector = ', embedding_vector)

print('words without embeddings =', words_without_embeddings)

# load the embedding matrix into a frozen layer
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

#-------------------------------------------------------------------
# command line arguments
conv1d_filter_count = int(sys.argv[1])
conv1d_filter_kernel_size = int(sys.argv[2])
max_pooling_window_size = int(sys.argv[3])
dense_size1 = int(sys.argv[4])
dense_size2 = int(sys.argv[5])
dropout_rate1 = float(sys.argv[6])
dropout_rate2 = float(sys.argv[7])
batch_size = int(sys.argv[8])
epoch_count = int(sys.argv[9])
print('filter-count = ', conv1d_filter_count, 'kernel-size=', conv1d_filter_kernel_size,
      'pooling-window=', max_pooling_window_size, 
      'dense-size1', dense_size1, 'dense-size2', dense_size2, 
      'dropout1=', dropout_rate1, 'dropout2=', dropout_rate2,
      'batch-size=', batch_size, 'epoch-count=', epoch_count)
#-------------------------------------------------------------------

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
x = Dense(dense_size1, activation='relu')(x)
x = Dropout(dropout_rate1)(x)
x = Dense(dense_size2, activation='relu')(x)
x = Dropout(dropout_rate2)(x)

preds = Dense(1, activation='sigmoid')(x)
print('shape dense = ',preds.shape)
model = Model(sequence_input, preds)

model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

class Metrics(CB.Callback):
    def __init__(self,x_train,y_train):
        self.training_X = x_train
        self.training_Y = y_train

    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.training_X)
        #print("pred=", pred)
        prediction = np.round(np.asarray(pred))
        target = self.training_Y
        precision = sklm.precision_score(target, prediction)
        recall = sklm.recall_score(target, prediction)
        f1_score = sklm.f1_score(target, prediction)
        avg_precision = sklm.average_precision_score(target,prediction,average='weighted')
        print("\nMetrics-train:",
              "precision=", precision, ",recall=", recall, 
              ",f1=", f1_score, ",avg_prec=", avg_precision)
    
        pred = self.model.predict(self.validation_data[0])
        prediction = np.round(np.asarray(pred))
        target = self.validation_data[1]
        precision = sklm.precision_score(target, prediction)
        recall = sklm.recall_score(target, prediction)
        f1_score = sklm.f1_score(target, prediction)
        avg_precision = sklm.average_precision_score(target,prediction,average='weighted')
        print("Metrics-val:", 
              "precision=", precision, ",recall=", recall, 
              ",f1=", f1_score, ",avg_prec=", avg_precision)

metrics = Metrics(x_train,y_train)

#------------------------------------------------------------------
# early stopping 
earlyStopping=CB.EarlyStopping(monitor='val_loss', patience=10, mode='auto')
# start learning
model.fit(x_train, y_train, validation_data=(x_val, y_val),
            epochs=epoch_count, batch_size=batch_size, callbacks=[metrics, earlyStopping], verbose=2)
print("Prediction on validation set:")
prediction = model.predict(x_val, batch_size, 1)
np.set_printoptions(threshold=np.inf)
# print the label against the rounded prediction for easy comparison
#print(np.column_stack((y_val, np.round(prediction))))

qi = 0
ai = 0
for pp in np.ndenumerate(prediction):
    if ai < validation_questions[qi].AnswerCount:
        validation_questions[qi].PredictedList.append((ai,pp))
        ai += 1
    else:
        ai = 0
        qi += 1

# Compute MAP
mean_average_precision = 0.0
for jj in range(len(validation_questions)):
    # Sort the answer prob from high to low
    validation_questions[jj].PredictedList.sort(key=lambda tup: tup[1], reverse=True)
    rank = 1
    # Iterate through predicted answers and find the correct one
    # aa = original answer index
    # bb = probability
    for (aa,bb) in validation_questions[jj].PredictedList:
        if aa == validation_questions[jj].CorrectAnswer:
            avg_prec = 1.0/float(rank) #avg prec = 1/rank_of_correct_answer
            print(jj,"Ap",avg_prec)
            mean_average_precision += avg_prec
        rank += 1

mean_average_precision /= float(len(validation_questions))

print("Mean avg Prec =", mean_average_precision)
