

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset = dataset.drop('Name', axis=1)
dataset = dataset.drop('Ticket', axis=1)
dataset = dataset.drop('PassengerId', axis=1)
dataset = dataset.drop('Fare', axis=1)
dataset = dataset.drop('Embarked', axis=1)
dataset = dataset.drop('Cabin', axis=1)

A = dataset.isnull().sum()

dataset['Age'].value_counts(dropna=False)
dataset['Age'].fillna(value=dataset['Age'].mean(), inplace=True)


#split
X = dataset.iloc[:, 1:6].values
y = dataset.iloc[:, 0].values


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# Fitting classifier to the Training set
# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X, y)



# Importing the dataset
dataset2 = pd.read_csv('test.csv')
dataset2= dataset2.drop('Name', axis=1)
dataset2 = dataset2.drop('Ticket', axis=1)
dataset2 = dataset2.drop('PassengerId', axis=1)
dataset2 = dataset2.drop('Fare', axis=1)
dataset2 = dataset2.drop('Embarked', axis=1)
dataset2 = dataset2.drop('Cabin', axis=1)



B = dataset2.isnull().sum()


dataset2['Age'].value_counts(dropna=False)
dataset2['Age'].fillna(value=dataset2['Age'].mean(), inplace=True)



Z = dataset2.iloc[:, 0:5].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Z[:, 1] = labelencoder.fit_transform(Z[:, 1])


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Z = sc.fit_transform(Z)

y_pred = classifier.predict(Z)


pd.DataFrame(y_pred).to_csv("Predictions_titanic.csv")




