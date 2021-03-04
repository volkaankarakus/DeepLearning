# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 21:28:36 2021

@author: VolkanKarakuş
"""

# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# standartization icin StandardScaler
from sklearn.preprocessing import StandardScaler
# gridsearch, KNN ile ilgili best parametreleri bulurken kullanicam.
from sklearn.model_selection import train_test_split, GridSearchCV
# cikan sonuclari gorsellestirmek icin en sonlarda kullanicam.
from matplotlib.colors import ListedColormap


# simdi cikan sonuclari degerlendirebilmem icin metrik kullanmam gerekiyor.
# basari yuzdesi icin accuracy_score,
from sklearn.metrics import accuracy_score, confusion_matrix
# confusion matrix de nerede hata yaptigima bakiyordum.

from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
# KNN uygulucam ve bunun component analizine bakicam. LocalOutlierFactor ile de outlierlarima bakicam.

from sklearn.decomposition import PCA

# warning library
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('breastCancerData.csv')
data.drop(['Unnamed: 32', 'id'], inplace=True, axis=1)

# diagnosis aslinda bizim target'imiz. Bunun ismini degistirelim.
data = data.rename(columns={'diagnosis': 'target'})

sns.countplot(data['target'])
print(data.target.value_counts())
# B    357
# M    212
# Name: target, dtype: int64

# %%
# target'in string degil int degerler olmasi gerekiyor.
# strip yapinca stringin basindaki boslugu atar. ' n' i 'n' yapar.
data['target'] = [1 if i.strip() == 'M' else 0 for i in data.target]

# %%
headData = data.head()
infoData = data.info()
describeData = data.describe()

# %% EDA

# elimizdeki veriler int ya da float turunde. correlation matrisine bakalim.
# Correlation
corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot=True, fmt='.2f')
plt.title('Correlation Between Features')
# bu grafikteki degerler Pearson katsayilari. Daha detayli daha sonra bak.
plt.show()

# %%
# grafik karisik oldugu icin correlated featurelarimiz icin ayri subsample correlation matrix cizdirelim.

threshold = -2.5
# -0.75 de olabilir, o yüzden mutlak deger alicam.
filtre = np.abs(corr_matrix['target']) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot=True, fmt='.2f')
plt.title('Correlation Between Features with Correlation Threshold=0.75')
# goruldugu uzere target'in bu 4 feature ile yuksek iliskiye sahip oldugunu gorebiliyoruz.

# box plot
# variable'i target olucak,var_name onemli degil,
data_melted = pd.melt(data, id_vars='target',
                      var_name='features', value_name='value')
# value_name

plt.figure()
sns.boxplot(x='features', y='value', hue='target', data=data_melted)
plt.xticks(rotation=60)
plt.show()

# %% pair plot(bizim numeric veriler icin en etkili yontemlerden birisi)
# gorseller pek de birsey anlatmiyor cunku normalization yapilmadi.
# sadece correlated featurelara bakalim.
sns.pairplot(data[corr_features], diag_kind='kde', markers='+', hue='target')
plt.show()  # kuyruklar saga dogru akan bir Gaussa pozitif skewness.(kuyrugun tailinin saga dogru uzamasi)

"""
skewness
"""

# %% Outlier
y = data.target
x = data.drop(['target'], axis=1)
columns = x.columns.tolist()

clf = LocalOutlierFactor()
# y_pred burada ML'deki gibi degil. buradaki y_pred outlier olup olmadigi prediction'i. 1 ve -1 dondurur.
y_pred = clf.fit_predict(x)

# degerlerin negatif cikmasi normal. negatifini aldigimiz icin.
X_score = clf.negative_outlier_factor_

outlier_score = pd.DataFrame()
outlier_score['score'] = X_score

# %%
# radius_mean ile texture_mean'i THRESHOLDSUZ cizdirelim.
# plt.figure()
# plt.scatter(x.iloc[:,0],x.iloc[:,1],color='k',s=3,label='Data Points') # s=3 boyutunda. Bunu run edince hangisi outlier hangisi degil
#                                                                             # gozlemleyemiyorum.

# # hangisi outlier hangisi degil diye gozle gormek istiyorum.gorsellestirebilmek icin normalization yapalim.
# radius=(X_score.max()-X_score)/(X_score.max()-X_score.min())
# outlier_score['radius']=radius
# plt.scatter(x.iloc[:,0],x.iloc[:,1],s=1000*radius,edgecolors='r',facecolors='none',label='Outlier Scores')
#                                                                     #facecolors=noktacigin icinin rengi
# plt.legend()
# plt.show()

#%% threshold koyalim. Threshold modeli egittikten sonra tekrar tune edilecek.
filtre = outlier_score['score'] < threshold
outlier_index = outlier_score[filtre].index.tolist()

plt.figure()
plt.scatter(x.iloc[outlier_index, 0], x.iloc[outlier_index,1], color='blue', s=50, label='Outliers')
plt.scatter(x.iloc[:, 0], x.iloc[:, 1], color='k', s=3, label='Data Points')

radius = (X_score.max()-X_score)/(X_score.max()-X_score.min())
outlier_score['radius'] = radius
plt.scatter(x.iloc[:, 0], x.iloc[:, 1], s=1000*radius,
            edgecolors='r', facecolors='none', label='Outlier Scores')
# facecolors=noktacigin icinin rengi
plt.legend()
plt.show()

# %% Drop Outliers
# x'te .values yapmama gerek yok. hala dataframe olarak kalsin.
x = x.drop(outlier_index)
# y'dekileri ilerleyen zamanlarda kullanicagim icin series olarak kalmasini istemiyorum. numpy'a cevirdim.
y = y.drop(outlier_index).values

# %% Train test split
test_size = 0.3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 42)

#%% Normalization
scaler=StandardScaler()
x_train = scaler.fit_transform(x_train) # x_train datasina gore bir scaler tanimla, fit et, ve x_train'e gore transformunu al.
x_test = scaler.transform(x_test) # fit edilmeyecek ama transform edilecek. x_test'imi de x_train'e gore egitilmis scaler'imi x_test uzerinde uyguluyprum.

#%%
# daha once gorsellestiremedigimiz box_plot'u gorsellestirelim.
x_train_df=pd.DataFrame(x_train,columns=columns)
x_train_df['target']=y_train

# box plot
data_melted=pd.melt(x_train_df,id_vars='target',var_name='features',value_name='value')

plt.figure()
sns.boxplot(x='features',y='value',hue='target',data=data_melted)
plt.xticks(rotation=90)
plt.show()

# pair plot
sns.pairplot(x_train_df[corr_features],diag_kind='kde',markers='+',hue='target')
plt.show()


#%% KNN
# KNN algoritmasi training sureci yoktur. x_train'i zaten biz belirlemistik.
# KNN algoritmasi outlier sevmez.
# KNN Big Data icin sikintilidir.(ML algoritmalarinin geneli oyle)

knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
cm=confusion_matrix(y_test,y_pred)

accuracy=accuracy_score(y_test,y_pred)
score=knn.score(x_test,y_test) # accuracy ve score ayni cikicak sadece kaynaklarda farkli gosterimlerle veriliyor.Burada 2 ornegi de yaptik.

print('Score    :',score)
print('Basic KNN Accuracy :',accuracy)
print('Confusion Matrix   :',cm)

# Confusion Matrix'e bakarsak  [[107   1]    -> 108 iyi huyludan 107'sini dogru, 63 kotu huyludan 56'sini dogru bilmisim.
#                               [  7  56]]

# y_test ile 95%'lik accuracy elde ettik ama train verideki accuracy'ye bakmadik.
    # Bu ne demek : Bizim KNN modelimiz overfit mi underfit mi bunu bilmiyoruz demek.
    
#%% Choosing Best Parameters
def KNN_Best_Params(x_train,x_test,y_train,y_test):
    
    k_range=list(range(1,31))
    weight_options=['uniform','distance']
    print() # araya bir bosluk atalim.
    param_grid=dict(n_neighbors=k_range,weights=weight_options)
    
    knn=KNeighborsClassifier()
    grid=GridSearchCV(knn,param_grid,cv=10,scoring='accuracy') # cross validation 10 kere
    grid.fit(x_train,y_train)
    print('Best training score {} with parameteres {}'.format(grid.best_score_,grid.best_params_))
    print()
    
    knn=KNeighborsClassifier(**grid.best_params_) # grid'i kullanarak da yapabilirdik. Farkli bir yol olsun diye boyle yaptik.
    knn.fit(x_train,y_train) # KNeighborsClassifier'i knn olarak yaratmak zorunda degildim. grid.fit de yapabilirdim.
    
    y_pred_test=knn.predict(x_test)
    y_pred_train=knn.predict(x_train) # herhangi bir overfit ya da underfit var mi gorebilmek icin train'i de predict etmek istiyorum.
    
    cm_test=confusion_matrix(y_test,y_pred_test)
    cm_train=confusion_matrix(y_train,y_pred_train)
    
    acc_test=accuracy_score(y_test,y_pred_test)
    acc_train=accuracy_score(y_train,y_pred_train)
    print('Test score : {}, Train Score : {}'.format(acc_test,acc_train))
    print()
    print('CM Test :',cm_test)
    print('CM Train :',cm_train)
    
    return grid # en iyi parametrelere sahip grid variable'i return edelim.

grid=KNN_Best_Params(x_train,x_test,y_train,y_test)

# Best training score 0.9670512820512821 with parameteres {'n_neighbors': 4, 'weights': 'uniform'}

# Test score : 0.9590643274853801, Train Score : 0.9773299748110831

# CM Test : [[107   2]
#            [  5  57]]
# CM Train : [[248   0]
#             [  9 140]]

# Train Score, Test Score'dan yuksek cikmis.yani burada bir ezberleme soz konusu(overfitting var).
    # ezberi azaltmak icin Cross Validation yapmistik bunun uzerine ya Model Complexity'yi azalticam ya da Regularization yapmam gerek.
    
# Model Complexity'yi azaltalim:
    # hyperparametreden k=2 almistik(n_neighbors)
        # Accuracy de Basic KNN Accuracy : 0.9532163742690059 cikmisti.
            # bu accuracy test veri seti icindi.
    # n=2 en iyi parametre olmadigina gore bunu x_test'te degil x_train'de predict edelim.
 #%%   
# knn=KNeighborsClassifier(n_neighbors=2)
# knn.fit(x_train,y_train)
# y_pred=knn.predict(x_train)
# accuracy=accuracy_score(y_train,y_pred)  yukaridaki bu kodlari x_test yerine x_train ile degistirmem gerek.

# bu sayede x_train basarim dustu ama ezberi azalttigim icin x_test oranim artti.

#%% PCA
# PCA mumkun oldugu kadar bilgi tutarak verinin boyutunun azaltilmasini saglayan yontemdir.
# Belli basli bir power ya da zaman kisitimiz varsa; verinin de boyutu cok fazlaysa (feature sayisi) PCA ile belli basli featurelari azaltabiliriz.
# 2. nedeni de elimizde bir corr() matrisimiz varsa ; burada da belli basli featurelar varsa bu featurelar birbirleriyle correle ise:
    # bu featurelari nasil cikarticagimizi bilmiyorsak , bu featurelarin ortadan kalkmasini PCA ile saglayabiliriz.
    
# bu proje icin aslinda veri az ve feauture da az ama biz yinede PCA kullanalim.
# PCA kullaniminin bir diger nedeni de gorsellestirmedir.

# PCA'in amaci eigen_value ve eigen_vector'leri bulmaktir. !!!!!!!!!
# Veriyi farkli bir space'e tasicam(Istedigim boyutta)
    # Eigen Vector: Bizim yeni space'imizin direction'idir.
    # Eigen Value: Bu directiondaki magnitude'lardir.
    
# 2 Boyutlu bir veri dusunelim. x ve y eksenlerinde olsun.
# Once boyutlarin ortalamasini bulucam. x ekseni ortalamasi ve y ekseni ortalamasi. (x^ ve y^)
# Daha sonra x ekseninden ve y ekseninden bunlari cikartip kendilerine esitliyorum.(x-x^=x, y-y^=y)

# Boylece ortalamlarini cikardigim icin 0 merkezli hale getirdim.
# |             x                                   |             
# |           x                                     |
# |      xxxx                                       |            x   
# |     x xx                   verisini             |        xxx           haline getirmis oldum.
# |   x   x                                         |    x xx
# |______________                                   |x____________
#                                                  x
#                                               x   x

# hem x hem de y'yi degistirdim. Simdi bunlarin covariance'ina bakalim. Covariance: iki degiskenin birlikte ne kadar degistigini gosterir.
# cov(x,y)=  var(x)      cov(x,y)
#            cov(y,x)    var(y)    2x2'lik matrise esittir. cov(x,y)=cov(y,x)'tir.

# cov(x,y)=E[x,y]-E[x].E[y] = cov(y,x) seklinde de soylenebilir. E: Expectation

# var(x)=E[x^2]-((E[x])^2)

#%%
# daha iyi anlamak icin basit bir ornek yapalim.
a=[2.4,0.6,2.1,2,3,2.5,1.9,1.1,1.5,1.2]
b=[2.5,0.7,2.9,2.2,3.0,2.3,2.0,1.1,1.6,0.8]

a=np.array(a)
b=np.array(b)

plt.scatter(a,b)
#%%
a_m=np.mean(a)
b_m=np.mean(b)

a=a-a_m
b=b-b_m

plt.scatter(a,b)      # Boylece merkezi orijine tasimis olduk. (0 merkezli hale geldi)

# covariance matrisi bulalim.

cov_matrix=np.cov(a,b)
print('Cov matrix :', cov_matrix)
# Cov matrix : [[0.53344444 0.56411111]
#               [0.56411111 0.68988889]]

# eigen_value ve eigen_vectorleri bulabilmem icin numpy'dan yardim alicam.
from numpy import linalg as LA

w,v=LA.eig(cov_matrix)

print('Eigen Values:',w)
print('Eigen Vectors:',v)
# Eigen Values: [0.04215805 1.18117528]
# Eigen Vectors: [[-0.75410555 -0.65675324]
#                 [ 0.65675324 -0.75410555]]

#%%
"""
v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
"""

p1=v[:,1]
p2=v[:,0]

plt.scatter(a,b)
plt.plot([0,p1[0]],[0,p1[1]])
plt.show()
#%%
# bunun boyutunu buyutelim,
plt.scatter(a,b)
plt.plot([-2*p1[0],2*p1[0]],[-2*p1[1],2*p1[1]])
plt.show() # bu benim main component'im. Cunku eigen_value'su digerinden daha buyuk.

# digerini de buna dik ve daha kucuk cikicak.

#%% Projede PCA

# PCA ' e baslamadan once veriyi standardization yapmamiz gerek.
# Daha once yapmistik tekrar yapmamizin nedeni : PCA unsupervised learning bir algoritma
#                                                yani herhangi bir class labelina ihtiyac duymuyor.
#                                                Sadece x_train verimizi kullanmaktansa tum x inputlarini scale edip tumunu kullanicaz.

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

pca=PCA(n_components=2) # pca'in 2 tane component'ini kullanicam.
pca.fit(x_scaled)
x_reduced_pca=pca.transform(x_scaled)  # x'in icindeki 30 tane feature'i 2 taneye dusurdum.

# bunu daha iyi gorebilmek icin bir dataframe'in icine atalim.
pca_data=pd.DataFrame(x_reduced_pca,columns=['p1','p2'])

# bu dataframe'e bir target column ekleyelim ki gorsellestirirken kullanabiliyim. yani y'yi.
pca_data['target']=y

sns.scatterplot(x='p1',y='p2',hue='target',data=pca_data)
plt.title('PCA: p1 vs p2')
plt.show()
# 30 boyutlu veriyi PCA kullanarak 2 boyuta dusurmus oldum.

#%%
# grafikte mavilerin icinde turuncular var bunlari knn yaparsam yanlis bulmus olucam.
x_train_pca,x_test_pca,y_train_pca,y_test_pca=train_test_split(x_reduced_pca,y,test_size=test_size,random_state=42)
grid_pca=KNN_Best_Params(x_train_pca,x_test_pca,y_train_pca,y_test_pca)

# Best training score 0.9419230769230769 with parameteres {'n_neighbors': 9, 'weights': 'uniform'}

# Test score : 0.9239766081871345, Train Score : 0.947103274559194

# CM Test : [[103   6]
#            [  7  55]]
# CM Train : [[241   7]
#             [ 14 135]]

# score'larimiz dustu, bunun sebebi de az onceki grafikte turuncular icindeki maviler, ve maviler icindeki turuncular.

# bunu bulmak icin gorsellestirme yapalim.

# visualize 
cmap_light = ListedColormap(['orange',  'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

h = .05 # step size in the mesh
X = x_reduced_pca
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = grid_pca.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class classification (k = %i, weights = '%s')"
          % (len(np.unique(y)),grid_pca.best_estimator_.n_neighbors, grid_pca.best_estimator_.weights))


#%% NCA
# buraya kadar accuracy'yi dusurduk, ezberlemeyi engellemek icin.

# NCA ile simdi accuracy arttiricaz. (Neighborhood Component Analysis)
# rastgele bir distance metrigi belirlemek yerine, dogrusal donusumu bularak bu metrigi NCA kendisi ogreniyor.

# PCA'deki gibi 2 tane component'a dusurelim.
nca=NeighborhoodComponentsAnalysis(n_components=2,random_state=42)
# NCA, PCA gibi unsupervised learning algoritmasi degildir. fit ederken y'ye ihtiyac duyar.
nca.fit(x_scaled,y)
x_reduced_nca=nca.transform(x_scaled)
nca_data=pd.DataFrame(x_reduced_nca,columns=['p1','p2'])

nca_data['target']=y
sns.scatterplot(x='p1',y='p2',hue='target',data=nca_data)
plt.title('NCA: p1 vs p2')
plt.show() 
# burada da maviler icinde turuncu, turuncular icinde maviler var ama NCA bunlari dogru tahmin edicek.

#%%
x_train_nca,x_test_nca,y_train_nca,y_test_nca=train_test_split(x_reduced_nca,y,test_size=test_size,random_state=42)
grid_nca=KNN_Best_Params(x_train_nca,x_test_nca,y_train_nca,y_test_nca)

# Best training score 0.9873076923076922 with parameteres {'n_neighbors': 1, 'weights': 'uniform'}

# Test score : 0.9941520467836257, Train Score : 1.0

# CM Test : [[108   1]
#            [  0  62]]
# CM Train : [[248   0]
#             [  0 149]]

#%% visualize 
cmap_light = ListedColormap(['orange',  'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

h = .2 # step size in the mesh
X = x_reduced_nca
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = grid_nca.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class classification (k = %i, weights = '%s')"
          % (len(np.unique(y)),grid_nca.best_estimator_.n_neighbors, grid_nca.best_estimator_.weights))

# %% find wrong decision
knn = KNeighborsClassifier(**grid_nca.best_params_)
knn.fit(x_train_nca,y_train_nca)
y_pred_nca = knn.predict(x_test_nca)
acc_test_nca = accuracy_score(y_pred_nca,y_test_nca)
knn.score(x_test_nca,y_test_nca)

test_data = pd.DataFrame()
test_data["X_test_nca_p1"] = x_test_nca[:,0]
test_data["X_test_nca_p2"] = x_test_nca[:,1]
test_data["y_pred_nca"] = y_pred_nca
test_data["Y_test_nca"] = y_test_nca

plt.figure()
sns.scatterplot(x="X_test_nca_p1", y="X_test_nca_p2", hue="Y_test_nca",data=test_data)

diff = np.where(y_pred_nca!=y_test_nca)[0]
plt.scatter(test_data.iloc[diff,0],test_data.iloc[diff,1],label = "Wrong Classified",alpha = 0.2,color = "red",s = 1000)                                                                                                            
