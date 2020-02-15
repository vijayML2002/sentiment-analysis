import tensorflow as tf
import os
import random
import collections
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation,Dropout,Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam

training_x=[]
training_y=[]
train_data=[]

path='D:/studies/cse/ml/nlp/aclImdb/train/neg'
path1='D:/studies/cse/ml/nlp/aclImdb/train/pos'
path2='D:/studies/cse/ml/nlp/aclImdb/test/neg'
path3='D:/studies/cse/ml/nlp/aclImdb/test/pos'

for i in os.listdir(path):
    with open(path+'/'+i,'r',encoding='utf-8') as f:
        training_x.append(f.read())
        training_y.append(0)

for i in os.listdir(path1):
    with open(path1+'/'+i,'r',encoding='utf-8') as f:
        training_x.append(f.read())
        training_y.append(1)

for i in os.listdir(path2):
    with open(path2+'/'+i,'r',encoding='utf-8') as f:
        training_x.append(f.read())
        training_y.append(0)

for i in os.listdir(path3):
    with open(path3+'/'+i,'r',encoding='utf-8') as f:
        training_x.append(f.read())
        training_y.append(1)

for i in range(0,len(training_x)):
    train_data.append([training_x[i],training_y[i]])
    
training_data=random.sample(train_data,len(train_data))
train_counter=collections.Counter([w for train in training_data for w in train[0].split()])

print("TOTAL NUMBER OF TRAINING DATA : ",len(training_data)) #training examples
print("NUMBER OF MOST COMMON WORDS : ",len(train_counter)) #most common words
print("TOTAL NUMBER OF WORDS : ",len([w for train in training_data for w in train[0].split()])) #total number of words

def clean_text(text):
    text = text.lower()    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)    
    return text

clean_data=[]
clean_label=[]

for train in training_data:
	clean_data.append(clean_text(train[0]))
	clean_label.append(train[1])

print("CLEANING DATA SUCCESSFULL(NUMBER OF CLEAN DATA) : ",len(clean_data))

def tokenize(x):
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x),tokenizer

def pad(x,length=None):
    return pad_sequences(x,maxlen=length,padding='post')

def preprocess(x):
    preprocess_x,x_tk=tokenize(x)
    preprocess_x=pad(preprocess_x)
    return preprocess_x,x_tk

process_clean_data,process_clean_tokenizer=preprocess(clean_data)

print("PREPROCESSING THE DATA IS COMPLETED")

max_length_data=process_clean_data.shape[1]
vocab_size=len(process_clean_tokenizer.word_index)
print("MAXIMUM LENGTH OF THE SETENCES : ",max_length_data)
print("WORDS IN TOKENIZER : ",len(process_clean_tokenizer.word_index))

def model_final(input_shape, vocab_size):
    learning_rate = 0.003
    model = Sequential()
    model.add(Embedding(vocab_size, 256, input_length=2482,
                         input_shape=input_shape[1:]))
    model.add(Flatten())
    model.add(Dense(2,activation='softmax'))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model

checkpoint_path="D:/studies/cse/2020pro/sentiment analysis/cp.ckpt"
checkpoint_dir=os.path.dirname(checkpoint_path)

cp_callback=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1)

model=model_final(process_clean_data.shape,vocab_size)
#model.fit(process_clean_data,clean_label,batch_size=100,epochs=5,callbacks=[cp_callback],validation_split=0.2)
model.load_weights(checkpoint_path)
print("SINCE WEIGHTS ARE ALREADY SAVED - IMPORTING THE SAVED MODEL")
print('***********NOW TRY IT*********')
#print('THE PROCESS OF TRAINING IS COMPLETED')

def classify(x,model,tokenizer):
    x=[x]
    process_clean_data=tokenizer.texts_to_sequences(x)
    process_clean_data=pad(process_clean_data,2482)
    test=process_clean_data
    prediction=model.predict(test)
    if prediction[0][0]>0.5:
        print('bad')
    else:
        print('good')
