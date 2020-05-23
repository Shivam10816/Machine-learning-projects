
#importing libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading data from file

data=pd.read_csv("BreastCancerdata.csv",index_col=0)

#printitng 1st 15 rows of data
print("Columns in data \n\n")
print(data.columns)

print("\n\nFirst 15 rows in data\n\n")
print(data.head(15))

#dropiing column conatiaining null value
print("\n\nFirst 15 rows in data\n\n")
data=data.dropna(axis=1)
print(data.head(15))

#Diviing data in input and output

X=data.iloc[:,1:31]
Y=data.iloc[:,0]



#checking if it conatins Null value

print("Null values in data:-\n",data.isna().sum())

#counting Mallagent and begin in dignosis column

print("\n\n",data["diagnosis"].value_counts())

#replacing M with 1 and B with 0

data=data.replace(["M","B"],["1","0"])
print("\n\n",data["diagnosis"].head(15))

#checking if all dtypes are in same format
print("\n\n",data.dtypes)

#conerting dtype to float from object
data["diagnosis"]=data["diagnosis"].astype(float)

print("\n\n",data.dtypes)

#visualzing data using histogram

for i in data.columns:
    plt.hist(data[i])
    plt.xlabel(i)
    plt.ylabel("count")
    plt.show()

#transforming lables into sutaible data

from sklearn.preprocessing import LabelEncoder

labelencode_y=LabelEncoder()

y=labelencode_y.fit_transform(Y)

#spliting data in train test data as 70:30


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)

#feature Scsling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#using 4 different logistic model 1)logistic regression 2)Navey Bias 3)decision Tree 4)RandomForstClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

Y_pred={}
accuracy_scores={}
confu_matr={}


# 1) logistic regression

class_log=LogisticRegression(random_state=0)
class_log.fit(x_train,y_train)
Y_pred["LogisticRegression"]=class_log.predict(x_test)   
accuracy_scores["LogisticRegression"]=accuracy_score(y_test,Y_pred["LogisticRegression"])
cm=confusion_matrix(y_test,Y_pred["LogisticRegression"])
confu_matr["LogisticRegression"]=cm

# 2)GaussianNB

class_NB=GaussianNB()
class_NB.fit(x_train,y_train)
Y_pred["GaussianNB"]=class_NB.predict(x_test)
accuracy_scores["GaussianNB"]=accuracy_score(y_test,Y_pred["GaussianNB"])
cm=confusion_matrix(y_test,Y_pred["GaussianNB"])
confu_matr["GaussianNB"]=cm

# 3)Decision tree

class_dt=DecisionTreeClassifier(random_state=0,criterion="entropy")
class_dt.fit(x_train,y_train)
Y_pred["DecisionTreeClassifier"]=class_dt.predict(x_test)
accuracy_scores["DecisionTreeClassifier"]=accuracy_score(y_test,Y_pred["DecisionTreeClassifier"])
confu_matr["DecisionTreeClassifier"]=confusion_matrix(y_test,Y_pred["DecisionTreeClassifier"])

# 4)Random Forest Classifier

class_rf =RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)

class_rf.fit(x_train,y_train)

Y_pred["RandomForestClassifier"]=class_rf.predict(x_test)

accuracy_scores["RandomForestClassifier"]=accuracy_score(y_test,Y_pred["RandomForestClassifier"])

confu_matr["RandomForestClassifier"]=confusion_matrix(y_test,Y_pred["RandomForestClassifier"])


    

#checking whether removing some attribute from data will effect accuracy or not

#removing 9 attibutes  'perimeter_mean','perimeter_se','perimeter_worst','symmetry_worst','fractal_dimension_se','concave points_mean','smoothness_worst','texture_mean','fractal_dimension_worst'

data=data.drop(labels=['perimeter_mean','perimeter_se','perimeter_worst','symmetry_worst','fractal_dimension_se','concave points_mean','smoothness_worst','texture_mean','fractal_dimension_worst'],axis=1)

#Diviing data in input and output

X=data.iloc[:,1:23]
Y=data.iloc[:,0]

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)



x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)





# 1) logistic regression


class_log.fit(x_train,y_train)
Y_pred["LogisticRegression_2"]=class_log.predict(x_test)   
accuracy_scores["LogisticRegression_2"]=accuracy_score(y_test,Y_pred["LogisticRegression_2"])
cm=confusion_matrix(y_test,Y_pred["LogisticRegression_2"])
confu_matr["LogisticRegression_2"]=cm

# 2)GaussianNB


class_NB.fit(x_train,y_train)
Y_pred["GaussianNB_2"]=class_NB.predict(x_test)
accuracy_scores["GaussianNB_2"]=accuracy_score(y_test,Y_pred["GaussianNB_2"])
cm=confusion_matrix(y_test,Y_pred["GaussianNB_2"])
confu_matr["GaussianNB_2"]=cm

# 3)Decision tree


class_dt.fit(x_train,y_train)
Y_pred["DecisionTreeClassifier_2"]=class_dt.predict(x_test)
accuracy_scores["DecisionTreeClassifier_2"]=accuracy_score(y_test,Y_pred["DecisionTreeClassifier_2"])
confu_matr["DecisionTreeClassifier_2"]=confusion_matrix(y_test,Y_pred["DecisionTreeClassifier_2"])

# 4)Random Forest Classifier



class_rf.fit(x_train,y_train)

Y_pred["RandomForestClassifier_2"]=class_rf.predict(x_test)

accuracy_scores["RandomForestClassifier_2"]=accuracy_score(y_test,Y_pred["RandomForestClassifier_2"])

confu_matr["RandomForestClassifier_2"]=confusion_matrix(y_test,Y_pred["RandomForestClassifier_2"])

print("\n\nAccuracy scores\n\n ")

for i in accuracy_scores:
    print(i,":-",accuracy_scores[i])
print("\n\n")

print("\n\nConfusion  matrix \n\n")
for i in confu_matr:
   
    print(i,":-")
    print(confu_matr[i])
    

