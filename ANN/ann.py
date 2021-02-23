# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 18:40:57 2021

@author: VolkanKarakuş
"""

# ANN de logistic regression gibi ama farki :
    # ann'de input layer(x_train) ve output layer(y_head)'in arasinda Hidden Layer dedigimiz kavramlar var.
    # en azindan bir tane Hidden Layer oldugu zaman bunu Neural Network olarak tanimlamis oluyorum.
    
# Hidden Layer denmesinin sebebi inputlari gormemesi.

# Input Layer - Hidden Layer - Output Layer sirasiyla da olsa da bu 2 Layer olarak gecer.(Input Layer soylenmez.)

# Hidden Layer'a istedigimiz kadar Node koyabiliriz.(Genelde input layer'daki node sayisinin yarisi alinir.)

# Artificial Neural Network'te Sigmoid function yerine x_train'i tanh fonksiyonuna sokuyorum. Buradan hidden layer'i elde
            #ettikten sonra hidden layer'i sigmoid function'a sokuyorum ve outputu buluyorum.(y_head)
        # Sigmoid Function outputu 0 ile 1 arasinda sikistiriyordu. Ortalamasi 0.5 e sikistirir.
    # tanh 1 ve -1 arasinda sikistirir. Burada ortalama 0'a sikistirir. Bu daha optimum.
    # Bu fonksiyonlarin amaci datayi karmasiklastirmak. Ne kadar karmasiklastirirsam datayi o kadar iyi ogrenirim.

#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# load data set
x_l = np.load('X.npy')
Y_l = np.load('Y.npy')
img_size = 64
plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')

# Join a sequence of arrays along an row axis.
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)

# Then lets create x_train, y_train, x_test, y_test arrays
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])
print("X train flatten",X_train_flatten.shape)
print("X test flatten",X_test_flatten.shape)

x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)

x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T


#%%# Evaluating the ANN

from keras.wrappers.scikit_learn import KerasClassifier # (bildigimiz classifier)
from sklearn.model_selection import cross_val_score # datayi bolup farkli kisimlar icin farkli trainlerle hata ortalama scoru buluyordu.
from keras.models import Sequential # initialize neural network library. (Inıtıalize parametrelere deger verir w ve b'ye)
from keras.layers import Dense # build our layers library (Layerlari construct etmek icin kullaniliyor.)
def build_classifier():
    # hiddenlayer1 icin 8 tane node olsun. units bunu gosterir.
    # kernel_initializer= weightleri initialize eder.
    # aktivasyon fonksiyonum Relu olsun. 0'dan kucukken 0, daha sonra y=x grafigi.
    # input_dim= x_train'in shape'i .
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    # sigmoid en sonki activation function. output layer icin.
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # compile icin optimizasyon parametreleri tanimlayalim.loss'u bulmak icin yukaridaki parametre.
    # back propogation icin optimizer'i kullanmam lazim.normalde learning rate'i sabit tutarak gitmistik.(gradient decent)
    # simdi yine gradient decent yapicaz ama 'adam'i kullanicaz. Adaptive Momentum
    # adam'da belli bir momente adapte olarak learning rate degisir ve daha hizli ogrenir.
    # metrics de degerlendirme metrigi. Modeli degerlendirirken accuracy'yi kullanicam diyorum.
    
    return classifier

# epochs= number of iteration
classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))