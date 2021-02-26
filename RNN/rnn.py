# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:25:28 2021

@author: VolkanKarakuş
"""

# Sequence Models
#   Sirali modellerdir.
#   TIME SERIES MODELDIR.
#   Zamana bagli modellerdir(zamanla degisen) - Hava alanindaki yolcu sayisinin zamanla degismesi gibi.
#       Speech Recognition - Natural Language Process(NLP) - Music Generation
#           Apple Siri - Google's Voice Search

#%% RNN
# Temporal memory'leri vardir(cok uzagi hatirlayamazlar).
# Gecmisi hatirlayip; gecmisle gelecek arasinda bagdaslastirir ve gelecege predict edebilir (ANN'de bu yoktu).
# Bellege sahiptirler, short term memory bir onceki node'da olanlari hatirlarlar.
# RNN short term memory'ye sahip ama LSTM long term memory'ye sahip olabiliyor.
# RNN'i ANN ya da CNN'den ayiran daha once de belirttigimiz gibi memory. Mesela 'VOLKAN' diye bir stringimiz var.
#   ve biz 4. harfe geldik yani K. ANN'e sordugumuz zaman 4. harfi K olan bir kelimenin 5. harfi ne olabilir diye.
#   ANN bilemez cunku memory olmadigi icin onceki harfleri birlestiremez. AMA RNN tam olarak bunu birlestirebilir.

# Exploding Gradients : Gradientin cok buyuk olmasi durumu. Gereksiz yere belli weightlere onem kazandirir.
# Vanishing Gradients : Gradientin cok kucuk olmasi durumu. Yavas ogrenir.
# Gradient : Cost'a gore weightlerdeki  degisimdi.   

#%% 
# loading and preprocessing data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the training set
dataset_train=pd.read_csv('Stock_Price_Train.csv')
headData=dataset_train.head()
#        Date    Open    High     Low   Close      Volume
# 0  1/3/2012  325.25  332.83  324.97  663.59   7,380,500
# 1  1/4/2012  331.27  333.87  329.08  666.45   5,749,400
# 2  1/5/2012  329.83  330.75  326.89  657.21   6,590,300
# 3  1/6/2012  328.34  328.77  323.68  648.24   5,405,900
# 4  1/9/2012  322.04  322.29  309.46  620.76  11,688,800

# borsa acilisindan yani Open'dan devam edelim.
train=dataset_train.loc[:,['Open']].values # values yaparak array'e cevirdim. 

#%%
# Feature Scaling (Normalization)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
train_scaled=scaler.fit_transform(train) #fit, datami 0'la 1 arasina fit eder. transform ise bunu transform edip belli bir seye esitler.

plt.plot(train_scaled)
plt.show()

#%%

# creating data structure with 50 timesteps and 1 output.
X_train=[] # train_data'yi ikiye ayirmak zorundayim. x benim inputlarim, y de predict edecegim degerim.
Y_train=[]
timesteps=50
# 50 tane sample'i X_train'e aticam. Daha sonra 51'i tahmin edip Y_train'e aticam.sonra yine 50 sample x_traine yine bir tane y_train'e.

for i in range(timesteps,1258):
    X_train.append(train_scaled[i-timesteps:i,0])    # ?????????
    Y_train.append(train_scaled[i,0])                # ????????? Neden 0 ekledik.
    
X_train,Y_train=np.array(X_train),np.array(Y_train)

# Reshaping
X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1)) # keras 3D matris ile calistigi icin boyle yaptik.

#%% Create RNN Model

from keras.models import Sequential # sequential'i data structure olarak dusunebilirdim.
                                        #Dense,SimpleRNN ve Dropout'u topladigim bir Neural Network yapisi.(constructor)
from keras.layers import Dense # layer yapisi (Hidden Layer'lari kullandigimiz)

from keras.layers import SimpleRNN

from keras.layers import Dropout # overfitting yapmamizi engelleyen yapi. Droput, regularization yontemlerinden biri. 


#%%
# Initializing RNN
regressor=Sequential()

# Adding the first RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True, input_shape = (X_train.shape[1], 1)))#?*?*
    #units : node
regressor.add(Dropout(0.2))


# bir sonrakiler bir oncekini kullanacagi icin shape vermeme gerek yok.
# Adding a second RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1)) # bir tane outputum olucak bu outputtan once bir tane node ekliyorum.

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') # adam : adaptive momentum

# Fitting the RNN to the Training set
regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)

# Result :
    # Epoch 100/100
    # 38/38 [==============================] - 1s 26ms/step - loss: 0.0022
    
#%% activation function'u degistirelim, layer sayisini attiralim ve epoch ile oynayalim . loss degisiyor mu ona bakalim.
regressor=Sequential()

# Adding the first RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50,activation='relu', return_sequences = True, input_shape = (X_train.shape[1], 1)))
    #units : node
regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units = 50,activation='relu', return_sequences = True))
regressor.add(Dropout(0.1))

# Adding a third RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50,activation='relu', return_sequences = True))
regressor.add(Dropout(0.2))

#
regressor.add(SimpleRNN(units = 50,activation='relu', return_sequences = True))
regressor.add(Dropout(0.1))

#
regressor.add(SimpleRNN(units = 50,activation='relu', return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50))
regressor.add(Dropout(0.1))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, Y_train, epochs = 100, batch_size = 16)

# Epoch 100/100
# 76/76 [==============================] - 3s 38ms/step - loss: 0.0013

#%% Prediction and Visualization RNN Model

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Stock_Price_Test.csv')
headTest=dataset_test.head()

real_stock_price = dataset_test.loc[:, ["Open"]].values # numpy array'e .values ile cevirdik.

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - timesteps:].values.reshape(-1,1)
inputs = scaler.transform(inputs)  # min max scaler

X_test = []
for i in range(timesteps, 70):
    X_test.append(inputs[i-timesteps:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price) # normalize halinden gercek degerlere cevir.

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
# epoch = 250 daha güzel sonuç veriyor.
# Daha guzel tahminler icin LSTMs'e geciyoruz.