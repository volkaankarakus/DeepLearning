# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 23:03:50 2021

@author: VolkanKarakuş
"""

#%% LSTM (Long Short Term Memory)

# RNN'in ozellesmis bir yapisidir.
# LSTM'de RNN'e gore long term memory var.


# jupyter notebook'taki grafigin aciklamasi :
    # tanh kullanmamizin nedeni : Vanishing gradient vardi.(cok kucuk gradient oldugunda yavas ogreniyorduk)
        # tanh ile turev uzun bir sure 0'a yaklasmaz. boylece gradient 0'a yaklasmayacagi icin ogrenme yavaslamaz.
        
    # h(t-1) : bir onceki layerdan gelen output degerleri. Simdiki LSTM icin input olmus oluyor, o yuzden inputa sokuyoruz.
    # X(t) : normal inputum.
    # c(t) : new updated memory. c(t)'nin oldugu cizgi memory line. bir onceki LSTM unitten aldigim c(t-1), 
        # islemler sonrasinda c(t) olarak cikar  
    # h(t) : nodeumun outputu.
    # h(t-1) ile x(t) birlesmiyor. arada + yok. bunlar paralel iki yapi.
    
# 1) Forget Gate : Unutma kapim. input olarak h(t-1) ve x(t)'yi aliyor ve bir sigmoid layer'a sokuyor.
                    # sigmoid layer'im 1 se aktarmis oluyorum. Yani memory line'ima iletmis oluyorum.0'sa unutmus oluyorum.
                    
# 2) Input Gate : Hangi bilginin memory'de depolanip depolanmayacagina karar veriyor. x(t) ve h(t-1) im var.
                    # bunlar sigmoidden geciyor.tanh ile birlesiyor. sonra X'da memoryde birlesip birlesmeyecegine karar veriliyor.
                    # sigmoid 0'sa eklenmiyor, 1'se memory'ye gecip depolaniyor.
                    
# 3) Output Gate : Hangi bilginin output olup olmadigina karar veriyor.

#%% Loading and Visualizing Data
import numpy
import pandas as pd 
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

data = pd.read_csv('international-airline-passengers.csv',skipfooter=5) # son 2,3 ornekte sikinti oldugu icin atladik.
data.head()

dataset = data.iloc[:,1].values
plt.plot(dataset)
plt.xlabel("time")
plt.ylabel("Number of Passenger")
plt.title("international airline passenger")
plt.show() # artis ve azalislar belli araliklarla. buna SEASONAL deniyor.

#%% Preprocessing Data
# reshape
# change type
# scaling
# train test split
# Create dataset

dataset = dataset.reshape(-1,1)  # reshape yapmasam (142,) olarak gozukecekti.Artik (142,1)
dataset = dataset.astype("float32")

# scaling (hem hiz arttirir, hem de daha iyi sonuc verir.)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.50) # yarisini train, kalani da test yapalim. 
test_size = len(dataset) - train_size
train = dataset[0:train_size,:]
test = dataset[train_size:len(dataset),:]
print("train size: {}, test size: {} ".format(len(train), len(test)))

time_stemp = 10
dataX = []
dataY = []
for i in range(len(train)-time_stemp-1):
    a = train[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(train[i + time_stemp, 0])
trainX = numpy.array(dataX)
trainY = numpy.array(dataY)  

dataX = []
dataY = []
for i in range(len(test)-time_stemp-1):
    a = test[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(test[i + time_stemp, 0])
testX = numpy.array(dataX)
testY = numpy.array(dataY)  

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# Datamizi kerasa uygun olan forma sokuyoruz, 2boyuttan(60,10), 3 boyuta(60,1,10) dönüştürüyoruz. 
#  Datayi donustururken basit rnn de yaptığımız gibi  (60,10,1) yerine neden (60,1,10) formuna donusturdugumuz.
#   Bu da datadan gelen veriye gore sekil aldigi icin.

#%% Create LSTM Model

model = Sequential()
model.add(LSTM(10, input_shape=(1, time_stemp))) # 10 lstm neuron(block) . Bir tane layerda 10 tane LSTM blogu olsun.
model.add(Dense(1)) # dense'de burada activation function yok.
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=1)
# Epoch 50/50
# 60/60 [==============================] - 0s 911us/step - loss: 0.0010

# keras LSTM ve keras Dense yazip google'da arat. farkli parametrelere bak.

#%% Prediction and Visualization LSTM Model

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# Train Score: 15.89 RMSE
# Test Score: 56.41 RMSE boyle yuksek cikmasi RMSE ile ilgili. Aslinda kucuk hatalar bulduk.
#%%
# shifting train
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[time_stemp:len(trainPredict)+time_stemp, :] = trainPredict
# shifting test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(time_stemp*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
