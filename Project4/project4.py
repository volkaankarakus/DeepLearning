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

#%% Feature Engineering
# Skewness
# target dependant variable

# dagilimin sekline bakalim:
sns.distplot(data.target,fit=norm) # datanin target'inin dagilimina bakalim. fit=norm ile normal dagilima bakalim(siyah)

# mu ve sigma degerlerine bakalim.
(mu,sigma)=norm.fit(data['target'])
print('mu : {}, sigma : {}'.format(mu,sigma)) # mu : 23.514572864321607, sigma : 7.806159061274433

#%%
# Dagilimin ne kadar gauss ne kadar normal dagilim oldugunu Histograma bakarak yapabilirim.
# bir baska secenek de qq plot.
plt.figure()
stats.probplot(data['target'],plot=plt)
plt.show()

"""
uclarda carpiklik var.
"""
#%%
#skewnessligi azaltmak icin log transform yapalim.
data['target']=np.log1p(data['target']) 
plt.figure()
sns.distplot(data.target,fit=norm)

# mu ve sigma degerlerine bakalim.
(mu,sigma)=norm.fit(data['target'])
print('mu : {}, sigma : {}'.format(mu,sigma)) # mu : 1.4195010209477965, sigma : 0.0788133543086489

plt.figure()
stats.probplot(data['target'],plot=plt)
plt.show()
"""
bastaki skewness ve carpiklik azalmis gorunuyor.
"""

#%% feature - independent variable
skewed_features=data.apply(lambda x:skew(x.dropna())).sort_values(ascending=False)
skewness=pd.DataFrame(skewed_features,columns=['skewed'])

"""
Box Cox Transformation ile skewneslik duzeltilebilir.(arastir)
"""
#%% One Hot Encoding
# silindir ve origin one-hot encoding yapilicakti.

# silindir ve origin'i categorical hale getirelim.
data['Cylinders']=data['Cylinders'].astype(str)
data['Origin']=data['Origin'].astype(str)

data=pd.get_dummies(data)

#%% Split and Standardization
x=data.drop(['target'],axis=1)
y=data.target

test_size=0.9
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=42)

#Standardization
scaler=StandardScaler() # RobustScaler da yapabilirdik.
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test) # x_test'te fit edilmiyor. scaler'im x_traine gore fit edilmisti.

#%% Regression Model
#Linear Regression

lr=LinearRegression()
lr.fit(x_train,y_train)
print('LR coefficients : ',lr.coef_)
y_predicted_dummy=lr.predict(x_test) # test prediction'imizi kullanarak prediction'a bakiyoruz.

mse=mean_squared_error(y_test,y_predicted_dummy)
print('Linear Regression MSE: ',mse)

# LR coefficients :  [-9.92705759e-02 -1.05006197e-01 -2.28689229e-02 -4.99057559e-02
#                      4.34952012e-02 -5.72828846e-02  4.72375426e-02  2.42861287e-17
#                     -1.40612813e-02 -2.28840651e-02 -7.11143569e-03 -2.73464310e-02
#                                                                      3.45783731e-02]
# Linear Regression MSE:  0.020632204780133005

#%% Ridge Regression (L2)
# overfitting'i engelleyen bir yontemdir.
# Ridge'in amaci : Less Square Error + lambda*(slope^2)'ni minimize etmektir.

ridge=Ridge(random_state=42,max_iter=10000)
alphas=np.logspace(-4,0.5,30)

tuned_parameters=[{'alpha':alphas}]
n_folds=5

clf=GridSearchCV(ridge,tuned_parameters,cv=n_folds,scoring='neg_mean_squared_error',refit=True) # refit=True clf'yi tekrar tekrar kullanmak icin bir parametre.

clf.fit(x_train,y_train)
scores=clf.cv_results_['mean_test_score']
scores_std=clf.cv_results_['std_test_score']

print('Ridge Coef :',clf.best_estimator_.coef_)

ridge=clf.best_estimator_
print('Ridge best estimator: ',ridge)

y_predicted_dummy=clf.predict(x_test) # burada clf'yi kullanabilmemin sebebi refit'i yukarida True yapmis olmam.
# clf zaten en iyi parametreleri kullanarak testi gerceklestirecek.
mse=mean_squared_error(y_test,y_predicted_dummy)
print('Ridge MSE :',mse)
print('------------------------------------')


# Ridge Coef : [-0.05212312 -0.07776735 -0.05991309 -0.03342344  0.04667136 -0.04514603
#                0.05431853  0.         -0.01496667 -0.03679728 -0.01124734 -0.02097308
#                                                                             0.0333379 ]

"""
Ridge coefficient'larin anlamli olmasi icin Lasso'ya da bakmamiz gerekiyor.
"""
# Ridge best estimator:  Ridge(alpha=3.1622776601683795, max_iter=10000, random_state=42)
# Ridge MSE : 0.01819808356385476
#%%
# son olarak da alfa'nin score'a gore nasil degistigini gozlemleyelim.
plt.figure()
plt.semilogx(alphas,scores) # alpha'yi yaratirken logspace kullanmistim, cizdirirken de logspace kullanmak zorundayim.
plt.xlabel('alpha')
plt.ylabel('score')
plt.title('Ridge')

#%% ###################### LASSO #################################

# Lasso(L1), formul olarak Ridge'in aynisi sadece slope'un karesi degil, mutlak degeri.
# Burada feature selection olarak Lasso kullanilabilir, cunku Lasso'da gereksiz coefficientlara 0 degeri atanir.
#   Ama Ridge'de coefficientlar kücük de olsa 0 degildir.

# Lasso'nun bir diger avantaji da eger High Correlated featurelarimiz varsa, bunlarin icinden sadece en onemlisini kullanir, gerisini
#   kullanmaz.(avantaj)

# Lassoda yine Ridge'de oldugu gibi biraz bias var. (train ve test veri seti arasinda cizgi cektigi icin x=0 icin y'deki deger bias biraz yukari kayar)
#   overfitting'i Lasso da Ridge gibi engeller. Zaten regularization yontemlerinin temel amacı OVERFITTING ENGELLEMEK.

#%% Lasso Regression(L1)

lasso = Lasso(random_state=42, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{'alpha': alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, scoring='neg_mean_squared_error',refit=True)
clf.fit(x_train,y_train)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']

print("Lasso Coef: ",clf.best_estimator_.coef_)
lasso = clf.best_estimator_
print("Lasso Best Estimator: ",lasso)

y_predicted_dummy = clf.predict(x_test)
mse = mean_squared_error(y_test,y_predicted_dummy)
print("Lasso MSE: ",mse)
print("---------------------------------------------------------------")

# Lasso Coef:  [-0.03758778 -0.08757891 -0.0646061  -0.02803523  0.0491826  -0.03359315
#                0.071679    0.          0.         -0.01794883 -0.         -0.00408397
#                                                                            0.04250482]
"""
Lasso gereksiz olan feature'lara direkt 0 degerini atiyor.
"""
# Lasso Best Estimator:  Lasso(alpha=0.004893900918477494, max_iter=10000, random_state=42)
# Lasso MSE:  0.01752159477082249
#%%
plt.figure()
plt.semilogx(alphas, scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Lasso")

#%%
################################################ ELASTICNET #####################################################

# Enet'ler de yine least square error'u minimize etmeye calisiyor.
# Formulunde lse + lambda1*(slope^2) + lambda2*|slope|
# Lasso ve Ridge'in karisimi gibi birsey.

# ElasticNet, HighlyCorrelated Feauturelarda cok ise yariyor.(correlated featurelarin cikartilmasinda)

parametersGrid = {"alpha": alphas,
                  "l1_ratio": np.arange(0.0, 1.0, 0.05)}

eNet = ElasticNet(random_state=42, max_iter=10000)
clf = GridSearchCV(eNet, parametersGrid, cv=n_folds, scoring='neg_mean_squared_error', refit=True)
clf.fit(x_train, y_train)


print("ElasticNet Coef: ",clf.best_estimator_.coef_)
print("ElasticNet Best Estimator: ",clf.best_estimator_)


y_predicted_dummy = clf.predict(x_test)
mse = mean_squared_error(y_test,y_predicted_dummy)
print("ElasticNet MSE: ",mse)


"""
StandardScaler
    Linear Regression MSE:  0.020632204780133015
    Ridge MSE:  0.019725338010801216
    Lasso MSE:  0.017521594770822522
    ElasticNet MSE:  0.01749609249317252
RobustScaler:
    Linear Regression MSE:  0.020984711065869643
    Ridge MSE:  0.018839299330570554
    Lasso MSE:  0.016597127172690837
    ElasticNet MSE:  0.017234676963922273  
"""

#%%
############################## XGBOOST #######################

# XGBoost, buyuk karmasik verisetleri icin gelistirilmis bir algoritmadir.

parametersGrid = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], 
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500,1000]}

model_xgb = xgb.XGBRegressor()

clf = GridSearchCV(model_xgb, parametersGrid, cv = n_folds, scoring='neg_mean_squared_error', refit=True, n_jobs = 5, verbose=True)

clf.fit(x_train, y_train)
model_xgb = clf.best_estimator_

y_predicted_dummy = clf.predict(x_test)
mse = mean_squared_error(y_test,y_predicted_dummy)
print("XGBRegressor MSE: ",mse)

# %% Averaging Models
# averaging modellerle ortalama alip daha iyi bir sonuc elde etmeye calisicaz.

class AveragingModels():
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)  

# en iyi xgboost ve lasso ciktigi icin bu ikisini alalim.
averaged_models = AveragingModels(models = (model_xgb, lasso))
averaged_models.fit(x_train, y_train)

y_predicted_dummy = averaged_models.predict(x_test)
mse = mean_squared_error(y_test,y_predicted_dummy)
print("Averaged Models MSE: ",mse)

"""
StandardScaler:
    Linear Regression MSE:  0.020632204780133015
    Ridge MSE:  0.019725338010801216
    Lasso MSE:  0.017521594770822522
    ElasticNet MSE:  0.01749609249317252
    XGBRegressor MSE: 0.017167257713690008
    Averaged Models MSE: 0.016034769734972223
RobustScaler:
    Linear Regression MSE:  0.020984711065869643
    Ridge MSE:  0.018839299330570554
    Lasso MSE:  0.016597127172690837
    ElasticNet MSE:  0.017234676963922273
    XGBRegressor MSE: 0.01753270469361755
    Averaged Models MSE: 0.0156928574668921
"""
