# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 21:28:36 2021

@author: VolkanKarakuş
"""

#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # standartization icin StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV # gridsearch, KNN ile ilgili best parametreleri bulurken kullanicam.
from matplotlib.colors import ListedColormap # cikan sonuclari gorsellestirmek icin en sonlarda kullanicam.


# simdi cikan sonuclari degerlendirebilmem icin metrik kullanmam gerekiyor.
from sklearn.metrics import accuracy_score,confusion_matrix # basari yuzdesi icin accuracy_score, 
                                                                # confusion matrix de nerede hata yaptigima bakiyordum.
                                                                
from sklearn.neighbors import KNeighborsClassifier,NeighborhoodComponentsAnalysis,LocalOutlierFactor
    # KNN uygulucam ve bunun component analizine bakicam. LocalOutlierFactor ile de outlierlarima bakicam.
    
from sklearn.decomposition import PCA 

# warning library
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('breastCancerData.csv')
data.drop(['Unnamed: 32','id'],inplace=True,axis=1)

#diagnosis aslinda bizim target'imiz. Bunun ismini degistirelim.
data=data.rename(columns={'diagnosis':'target'})

sns.countplot(data['target'])
print(data.target.value_counts())
# B    357
# M    212
# Name: target, dtype: int64

#%%
# target'in string degil int degerler olmasi gerekiyor.
data['target']=[1 if i.strip()=='M' else 0 for i in data.target] # strip yapinca stringin basindaki boslugu atar. ' n' i 'n' yapar.

#%%
headData=data.head()
infoData=data.info()
describeData=data.describe()

#%% EDA

# elimizdeki veriler int ya da float turunde. correlation matrisine bakalim.
# Correlation
corr_matrix=data.corr()
sns.clustermap(corr_matrix,annot=True,fmt='.2f')
plt.title('Correlation Between Features')
plt.show() # bu grafikteki degerler Pearson katsayilari. Daha detayli daha sonra bak.

#%% 
# grafik karisik oldugu icin correlated featurelarimiz icin ayri subsample correlation matrix cizdirelim.

threshold=0.75
filtre=np.abs(corr_matrix['target']) > threshold # -0.75 de olabilir, o yüzden mutlak deger alicam.
corr_features=corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(),annot=True,fmt='.2f')
plt.title('Correlation Between Features with Correlation Threshold=0.75')
# goruldugu uzere target'in bu 4 feature ile yuksek iliskiye sahip oldugunu gorebiliyoruz.

#box plot
data_melted=pd.melt(data,id_vars='target',var_name='features',value_name='value') # variable'i target olucak,var_name onemli degil,
                                                                                    #value_name

plt.figure()
sns.boxplot(x='features',y='value',hue='target',data=data_melted)
plt.xticks(rotation=60)
plt.show()

#%% pair plot(bizim numeric veriler icin en etkili yontemlerden birisi)
# gorseller pek de birsey anlatmiyor cunku normalization yapilmadi.
sns.pairplot(data[corr_features],diag_kind='kde',markers='+',hue='target') # sadece correlated featurelara bakalim.
plt.show() # kuyruklar saga dogru akan bir Gaussa pozitif skewness.(kuyrugun tailinin saga dogru uzamasi)

"""
skewness
"""
                                                                                                                                
