
import sys
#sys.path.append("C:/Users/pauli/AppData/Local/Programs/Python/Python38/site")

import warnings
warnings.simplefilter('ignore')
import numpy as np 
import pandas as pd 
import scipy.stats as stats
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn

from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix  # for model evaluation
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


#read data
data = pd.read_csv("Ratings&Fundamentals.csv")
data.head()
len(data)

#plot some scatter graphs
plt.scatter(data.fcf, data.OurRating, c='g', s=4)

plt.scatter(data.eps, data.OurRating, c = 'r', s=4)

plt.scatter(data.roe, data.OurRating, c = 'b', s=4)

plt.scatter(data.roa, data.OurRating, c = 'y', s=4)

plt.scatter(data.ebit, data.OurRating, c = 'b', s=4)

plt.scatter(data.OurRating, data.netinc, c = 'orange', s=4)

#set numerical values to OurRating
data_new = data
data_new['OurRating'] = data['OurRating'].apply(
        lambda x: 1 if (x == "Not Junk") else 0)

df = data_new.loc[:,'OurRating':'fcfps']
df

#These rows had NAN values, hence I replaced the missing values with 0
#data.iloc[[747]]
#data.iloc[[748]]
#data.iloc[[752]]

import matplotlib
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

cor_target = abs(cor["OurRating"])

#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.2]
relevant_features

print(df[["netmargin","equity"]].corr())
print(df[["netmargin","opinc"]].corr())
print(df[["equity","opinc"]].corr())

X = df.drop("OurRating",1)
y = df["OurRating"]

## Recursive Feauture Elimination (RFE)

#no of features
nof_list=np.arange(1,30)            
high_score=0

#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

cols = list(X.columns)
model = LinearRegression()

#Initializing RFE model
rfe = RFE(model, 10)             

#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  

#Fitting the data to model
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)

## Backward Elimination

X_1 = sm.add_constant(X)


#Fitting sm.OLS model
model = sm.OLS(y,X_1).fit()
model.pvalues

cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)


## LASSO
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")

## LDA
train, test = train_test_split(df, train_size = 0.4)
X = train.loc[:, ['ev', "netmargin"]]
y = train.loc[:, ['OurRating']]
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
lda.predict(test.loc[:, ['ev', "netmargin"]])
pred = lda.predict(test.loc[:, ['ev', "netmargin"]])
true = test['OurRating'].values
confusion_matrix(true, pred)

# QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X, y)

pred = qda.predict(test.loc[:, ['ev', "netmargin"]])
true = test['OurRating'].values
confusion_matrix(true, pred)