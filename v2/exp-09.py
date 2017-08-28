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

EMBEDDING_DIM = 100
BASE_DIR = '/home/indy/data'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
EMBEDDINGS_FILE = 'glove.6B.' + str(EMBEDDING_DIM) + 'd.txt'
WANG_DATA_DIR = BASE_DIR + '/wang/'
MAX_SEQUENCE_LENGTH = 70
MAX_NB_WORDS = 40000
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
            ################
            #DEBUG
            #if len(qs.items()) > 100:
            #    break
            ################
    nb_questions = len(qs.items())
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

# load the embedding matrix into a frozen layer
embedding_layer = Embedding(len(vocabulary) + 1,
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

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
print('shape input = ', sequence_input.shape)
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
    def __init__(self,x_tr,y_tr):
        self.training_X = x_tr
        self.training_Y = y_tr

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

metrics = Metrics(x_tr,y_tr)

#-------------------------------------------------------------------
# early stopping 
earlyStopping=CB.EarlyStopping(monitor='val_loss', patience=5, mode='auto')
ttt1 = time.time()
print("Time taken for prep:",ttt1-ttt0,"s")
# start learning
model.fit(x_tr, y_tr, validation_data=(x_va, y_va),
          epochs=epoch_count, 
          batch_size=batch_size, 
          #callbacks=[metrics, earlyStopping], 
          callbacks=[earlyStopping], 
          verbose=2)
np.set_printoptions(threshold=np.inf)
batch_size=1
print("\nPrediction on test set:")
te_predictions = model.predict(x_te, batch_size, 1)
print("\nPrediction on validation set:")
va_predictions = model.predict(x_va, batch_size, 1)
#print("VA_PREDICTIONS:",va_predictions)
print("\nPrediction on training set:")
tr_predictions = model.predict(x_tr, batch_size,1)

print("Prediction len(x_te) - len(te_prediction)",len(x_te),len(te_predictions))

# print the label against the rounded prediction for easy comparison
#print(np.column_stack((y_va, np.round(prediction))))

def ComputeStats(prediction,target):
    tp = 0; fp = 0; tn = 0; fn = 0

    for i in range(len(prediction)):
        if target[i] == 1:
            if prediction[i] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if prediction[i] == 0:
                tn += 1
            else:
                fp += 1
    return(tp,fp,tn,fn)

print("\n\n")

tp,fp,tn,fn = ComputeStats(np.round(te_predictions), y_te)
print("TEST tp=",tp,"fp=",fp,"tn=",tn,"fn=",fn)
tp,fp,tn,fn = ComputeStats(np.round(va_predictions), y_va)
print("VALIDATION tp=",tp,"fp=",fp,"tn=",tn,"fn=",fn)
#tp,fp,tn,fn = ComputeStats(np.round(tr_predictions), y_tr)
tp=0;fp=0;tn=0;fn=0;
print("TRAINING tp=",tp,"fp=",fp,"tn=",tn,"fn=",fn)

# Comutes the Mean Average Precision of the prediction
# qas is a list of QuestionAnswerSet
# prediction is a list of probabilities (floating point) how likely
# this answer is correct. We fed in samples of q+a to the model. 
def ComputeMAP(qas,prediction,qii,print_stats=False):
    qi = 0  # Question index
    ai = 0  # Answer index
    k=0
    for _,pp in np.ndenumerate(prediction):
        qi = qii[k]
        k+=1
        qas[qi].PredictedList.append((ai,pp))
        ai += 1
        if ai == qas[qi].AnswerCount:
             ai = 0

    # Compute MAP
    mean_average_precision = 0.0
    for jj in range(len(qas)):
        # Sort the answer prob from high to low
        #print("Before shuffle",qas[jj].PredictedList)
        np.random.shuffle(qas[jj].PredictedList)
        #print("After shuffle",qas[jj].PredictedList)
        qas[jj].PredictedList.sort(key=lambda tup: tup[1], reverse=True)
        rank = 1
        if print_stats:
            #print("Q",qas[jj].QuestionIndex,"correct answers=",len(qas[jj].CorrectAnswers),"/",qas[jj].AnswerCount)
            print("QID",qas[jj].QuestionIndex,"CorrectAnswers",qas[jj].CorrectAnswers,"/",qas[jj].AnswerCount,"PredictedList",qas[jj].PredictedList)
        # Iterate through predicted answers and find the correct one
        # ai = original answer index
        # pp = probability
        for (ai,pp) in qas[jj].PredictedList:
            if ai in qas[jj].CorrectAnswers:
                avg_prec = 1.0/float(rank) #avg prec = 1/rank_of_correct_answer
                #print(jj,"Ap",avg_prec)
                mean_average_precision += avg_prec
                # Since there can be multiple correct answers, we only add the first hit
                break
            rank += 1

    mean_average_precision /= float(len(qas))
    return mean_average_precision

print("\n\n\n\nComputing MAP for validation set")
va_map = ComputeMAP(va_questions, va_predictions, va_qii, True)
print("\n\n\n\nComputing MAP for test set")
te_map = ComputeMAP(te_questions, te_predictions, te_qii, True)
tr_map = 0.0
#print("Computing MAP for training set")
#tr_map = ComputeMAP(tr_questions, tr_predictions, tr_qii)

print("Test Mean avg Prec =", te_map)
print("Val Mean avg Prec =", va_map)
print("Train Mean avg Prec =", tr_map)
