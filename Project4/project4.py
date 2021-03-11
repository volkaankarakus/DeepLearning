# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:19:38 2021

@author: VolkanKarakuş
"""

#%% 
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm,skew

from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin,clone # bunlari averaging model'de yani tum 
                                                                                # modelleri birlestirirken kullanicaz.
# XGBoost 
# import xgboost as xgb

# warning
import warnings
warnings.filterwarnings('ignore')

column_name=['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year','Origin']
data=pd.read_csv('auto-mpg.data',names=column_name,na_values='?',comment='\t',sep=' ',skipinitialspace=True)

#mpg'yi target olarak degistiricem, cunku bu dependant.
data=data.rename(columns={'MPG':'target'})

infoData=data.info() # horse power icerisinde 6 tane NaN value var.

describeData=data.describe()

#%% Missing Values
print(data.isna().sum())
# target          0
# Cylinders       0
# Displacement    0
# Horsepower      6
# Weight          0
# Acceleration    0
# Model Year      0
# Origin          0
# dtype: int64

data['Horsepower']=data['Horsepower'].fillna(data['Horsepower'].mean())
sns.distplot(data.Horsepower)

#%% EDA
corr_matrix=data.corr()
sns.clustermap(corr_matrix,annot=True,fmt='.2f')
plt.title('Correlation between features')
plt.show()

#%%
threshold=0.75
filtre=np.abs(corr_matrix['target'])>threshold
corr_features=corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(),annot=True,fmt='.2f')
plt.title('Correlation between features')
plt.show()

"""
grafikte birbirleriyle yuksek iliskili featurelar varsa bunlar birbirleriyle esduzlemlidir.
yani MULTICOLLINEARITY.
Bu bizim avantaj degil. Dezavantaj olmasinin sebebi bir feature kullanmak yerine daha fazla featurela ugrasmak bizim modelimizi
    yanlis yonlendirebilir. 
"""
#%%
sns.pairplot(data,diag_kind='kde',markers='+')
plt.show()
"""
bu grafikte targetta pozitif skewness var.
    silindirlerde 4,6,8. cok az 3 ve 5 var.Silindiri categorical dusunup feature extraction'da kullanabiliriz.
    Origin de 3 tane veriye sahip oldugu icin categorical dusunulebilir.
alt ucgenle ust ucgen birbirinin aynisi. Daha fazla iliski yok gibi duruyor.
"""

#%% bunlara daha detayli bakalim.
plt.figure()
sns.countplot(data['Cylinders'])
print(data['Cylinders'].value_counts())

plt.figure()
sns.countplot(data['Origin'])
print(data['Origin'].value_counts())

#%% boxplot
for c in data.columns:
    plt.figure()
    sns.boxplot(x=c,data=data,orient='v') # ortaya cikacak oryantasyonlar vertical olsun.
    
"""
Acceleration ve horsepower'da outlierlar var. 
Bunlari cikarticaz.
"""

#%%

thr=2 # carpimda genelde 1.5 alinir ama biz bu grafik icin 2 alalim.

horsepower_desc=describeData['Horsepower']
q3_hp=horsepower_desc[6]
q1_hp=horsepower_desc[4]
IQR_hp=q3_hp-q1_hp
top_limit_hp=q3_hp + thr*IQR_hp
bottom_limit_hp=q1_hp - thr*IQR_hp
filter_hp_bottom=bottom_limit_hp < data['Horsepower']
filter_hp_top=data['Horsepower'] < top_limit_hp
filter_hp=filter_hp_bottom & filter_hp_top

data=data[filter_hp] 

acceleration_desc = describeData["Acceleration"]
q3_acc = acceleration_desc[6]
q1_acc = acceleration_desc[4]
IQR_acc = q3_acc - q1_acc # q3 - q1
top_limit_acc = q3_acc + thr*IQR_acc
bottom_limit_acc = q1_acc - thr*IQR_acc
filter_acc_bottom = bottom_limit_acc < data["Acceleration"]
filter_acc_top= data["Acceleration"] < top_limit_acc
filter_acc = filter_acc_bottom & filter_acc_top

data = data[filter_acc] # remove Horsepower outliers
