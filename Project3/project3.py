# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:07:11 2021

@author: VolkanKarakuş
"""

#%% SENTIMENT ANALYSIS

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import SimpleRNN, Dense, Activation

(x_train,x_test),(y_train,y_test)=imdb.load_data(path='imdb.npz', # daha once yuklenmedigi icin path ile keras'in icerisindeki datasetlere indir.npz, numpy'in zip'li hali gibi dusunebiliriz.
                                       num_words=None , # num_words=None; butun kelimeleri al demek. num_words=100 olsaydi en cok kullanilan 100 kelimeyi al olucakti.
                                       skip_top=0     , # en sik kullanilan kelimeleri ignore etme(the,and gibi.)
                                       maxlen=None    , # yorum uzunlugu default kalsin herhangi bir cropping yapmayalim.
                                       seed = 113     , # seed=113 icin kerasin kendi sitesinde uygun bir parametre(datalarin shuffle ile ilgili)
                                       start_char=1   , # yine kerasin bir parametresi
                                       oov_char=2     , # " "            "  "
                                       index_from=3   )
#%%
print('type of x_train : {}'.format(type(x_train))) # type of x_train : <class 'numpy.ndarray'>

print('shape of x_train : {}'.format(x_train.shape)) # shape of x_train : (25000,)
print('shape of x_test : {}'.format(x_test.shape)) # shape of x_train : (25000,)

"""
x_train'in her bir elemaninin icinde aslinda kelimeler var : birine ornek yapalim
"""

d=x_train[0] # d'nin icinde 218 kelime , integer'a kodlanmis sekilde .

#%% EDA

print("Y train values: ",np.unique(y_train))
print("Y test values: ",np.unique(y_test))

unique, counts = np.unique(y_train, return_counts = True)

unique, counts = np.unique(y_test, return_counts = True)
#%%
plt.figure()
sns.countplot(y_train)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("Y train")

plt.figure()
sns.countplot(y_test)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("Y test")


review_len_train = []
review_len_test = []
for i, ii in zip(x_train, x_test):
    review_len_train.append(len(i))
    review_len_test.append(len(ii))

sns.distplot(review_len_train, hist_kws = {"alpha":0.3})
sns.distplot(review_len_test, hist_kws = {"alpha":0.3})

print("Train mean:", np.mean(review_len_train))
print("Train median:", np.median(review_len_train))
print("Train mode:", stats.mode(review_len_train))

#%% number of words
word_index = imdb.get_word_index() # get_word_index yapmamin sebebi kelimeler degil, kelime degerlerinin int olarak verilmesi.
                                        # bu nedenle index aliyorum.
print(type(word_index))
print(len(word_index))

for keys, values in word_index.items(): # bu bir dictionary oldugu icin .items() ile keys ve values ile for loopta dondurebilirim.
    if values == 22:
        print(keys)

def whatItSay(index = 24):          # ???????????????????????????????????????????????????????????????????????**
    
    reverse_index = dict([(value, key) for (key, value) in word_index.items()])
    decode_review = " ".join([reverse_index.get(i - 3, "!") for i in x_train[index]])
    print(decode_review)
    print(y_train[index])
    return decode_review

decoded_review = whatItSay(36)

# %% Preprocess

num_words = 15000
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=num_words)

maxlen = 130
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

print(X_train[5])

for i in X_train[0:10]:
    print(len(i))

decoded_review = whatItSay(5)


# %% RNN

rnn = Sequential()
rnn.add(Embedding(num_words, 32, input_length = len(X_train[0])))
rnn.add(SimpleRNN(16, input_shape = (num_words,maxlen), return_sequences= False, activation= "relu"))
rnn.add(Dense(1))
rnn.add(Activation("sigmoid"))

print(rnn.summary())
rnn.compile(loss = "binary_crossentropy", optimizer="rmsprop",metrics= ["accuracy"])

history = rnn.fit(X_train, Y_train, validation_data= (X_test, Y_test), epochs=5, batch_size= 128, verbose=1)

score = rnn.evaluate(X_test, Y_test)
print("Accuracy: %",score[1]*100)

plt.figure()
plt.plot(history.history["acc"], label = "Train")
plt.plot(history.history["val_acc"], label = "Test")
plt.title("Acc")
plt.ylabel("Acc")
plt.xlabel("Epochs")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], label = "Train")
plt.plot(history.history["val_loss"], label = "Test")
plt.title("Acc")
plt.ylabel("Acc")
plt.xlabel("Epochs")
plt.legend()
plt.show()
