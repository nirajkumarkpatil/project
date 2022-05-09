import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as se
df=pd.read_csv('lr_seattle-weather.csv')
pd1=df.drop(columns=['date'])
pd1=pd1.dropna()


print("Old Shape: ", pd1.shape)
upper = np.where((pd1['precipitation'].mean)()+3*pd1['precipitation'].std()<pd1['precipitation'])
lower = np.where((pd1['precipitation'].mean)()-3*pd1['precipitation'].std()>pd1['precipitation']) 
pd1.drop(upper[0], inplace = True)
pd1.drop(lower[0], inplace = True) 
print("New Shape: ", pd1.shape)
print(pd1.isnull().sum())

from sklearn.preprocessing import LabelEncoder,OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression, LogisticRegression
#from sklearn.naive_bayes import GaussianNB
#from sklearn.neighbors import KNeighborsClassifier, KNeighborsReegressor
#from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

X=pd1[['precipitation','temp_max','temp_min','wind']]
y=pd1['e']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.20)
#print(X_train.shape)
#print(X_test.shape)


from sklearn.preprocessing import RobustScaler
rosc=RobustScaler()
X_train=rosc.fit_transform(X_train)
X_test=rosc.fit_transform(X_test)


le=LabelEncoder()
y_train=le.fit_transform(y_train)
y_test=le.fit_transform(y_test)
#print(y_train)
#print(y_test)

treee=tree.DecisionTreeClassifier()
treee.fit(X_train, y_train)
print(treee.score(X_train, y_train))

import pickle
with open('my_model.pkl','wb') as first:
    pickle.dump(treee, first)
    
with open('my_model.pkl','rb') as first:
    model=pickle.load(first)
    

