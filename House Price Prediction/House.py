

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset = dataset.drop('Id', axis=1 )
dataset = dataset.drop('Alley', axis=1)
dataset = dataset.drop('PoolQC', axis=1)
dataset = dataset.drop('Fence', axis=1)
dataset = dataset.drop('MiscFeature', axis=1)
dataset = dataset.drop('FireplaceQu', axis=1)
dataset = dataset.drop('MasVnrType', axis=1)
dataset = dataset.drop('GarageYrBlt', axis=1)
dataset = dataset.drop('Electrical', axis=1)


# taking care of missing data
A = dataset.isnull().sum()

dataset['BsmtQual'].value_counts(dropna=False)
dataset['BsmtQual'].fillna(value='TA', inplace=True)
dataset['BsmtCond'].value_counts(dropna=False)
dataset['BsmtCond'].fillna(value='TA', inplace=True)
dataset['BsmtExposure'].value_counts(dropna=False)
dataset['BsmtExposure'].fillna(value='No', inplace=True)
dataset['GarageType'].value_counts(dropna=False)
dataset['GarageType'].fillna(value='Attchd', inplace=True)
dataset['GarageFinish'].value_counts(dropna=False)
dataset['GarageFinish'].fillna(value='RFn', inplace=True)
dataset['GarageQual'].value_counts(dropna=False)
dataset['GarageQual'].fillna(value='TA', inplace=True)
dataset['GarageCond'].value_counts(dropna=False)
dataset['GarageCond'].fillna(value='TA', inplace=True)
dataset['BsmtFinType1'].value_counts(dropna=False)
dataset['BsmtFinType1'].fillna(value='ALQ', inplace=True)
dataset['BsmtFinType2'].value_counts(dropna=False)
dataset['BsmtFinType2'].fillna(value='Unf', inplace=True)
dataset['MasVnrArea'].value_counts(dropna=False)
dataset['MasVnrArea'].fillna(value=0.0, inplace=True)

dataset['LotFrontage'].mean()
dataset['LotFrontage'].value_counts(dropna=False)
dataset['LotFrontage'].fillna(value=dataset['LotFrontage'].mean(), inplace=True)


#split
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 71].values




# Encoding categorical data


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])
X[:, 4] = labelencoder.fit_transform(X[:, 4])
X[:, 5] = labelencoder.fit_transform(X[:, 5])
X[:, 6] = labelencoder.fit_transform(X[:, 6])
X[:, 7] = labelencoder.fit_transform(X[:, 7])
X[:, 8] = labelencoder.fit_transform(X[:, 8])
X[:, 9] = labelencoder.fit_transform(X[:, 9])
X[:, 10] = labelencoder.fit_transform(X[:, 10])
X[:, 11] = labelencoder.fit_transform(X[:, 11])
X[:, 12] = labelencoder.fit_transform(X[:, 12])
X[:, 13] = labelencoder.fit_transform(X[:, 13])
X[:, 14] = labelencoder.fit_transform(X[:, 14])
X[:, 19] = labelencoder.fit_transform(X[:, 19])
X[:, 20] = labelencoder.fit_transform(X[:, 20])
X[:, 21] = labelencoder.fit_transform(X[:, 21])
X[:, 22] = labelencoder.fit_transform(X[:, 22])
X[:, 24] = labelencoder.fit_transform(X[:, 24])
X[:, 25] = labelencoder.fit_transform(X[:, 25])
X[:, 26] = labelencoder.fit_transform(X[:, 26])
X[:, 27] = labelencoder.fit_transform(X[:, 27])
X[:, 28] = labelencoder.fit_transform(X[:, 28])
X[:, 29] = labelencoder.fit_transform(X[:, 29])
X[:, 30] = labelencoder.fit_transform(X[:, 30])
X[:, 32] = labelencoder.fit_transform(X[:, 32])
X[:, 36] = labelencoder.fit_transform(X[:, 36])
X[:, 37] = labelencoder.fit_transform(X[:, 37])
X[:, 38] = labelencoder.fit_transform(X[:, 38])
X[:, 49] = labelencoder.fit_transform(X[:, 49])
X[:, 51] = labelencoder.fit_transform(X[:, 51])
X[:, 54] = labelencoder.fit_transform(X[:, 54])
X[:, 53] = labelencoder.fit_transform(X[:, 53])
X[:, 58] = labelencoder.fit_transform(X[:, 58])
X[:, 59] = labelencoder.fit_transform(X[:, 59])
X[:, 57] = labelencoder.fit_transform(X[:, 57])
X[:, 70] = labelencoder.fit_transform(X[:, 70])
X[:, 69] = labelencoder.fit_transform(X[:, 69])


##onehotencoder = OneHotEncoder(categorical_features = [3])
## X = onehotencoder.fit_transform(X).toarray()


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)








# for test set results
# test dataset

test_dataset = pd.read_csv('test.csv')
test_dataset = test_dataset.drop('Alley', axis=1)
test_dataset = test_dataset.drop('PoolQC', axis=1)
test_dataset = test_dataset.drop('Fence', axis=1)
test_dataset = test_dataset.drop('MiscFeature', axis=1)
test_dataset = test_dataset.drop('FireplaceQu', axis=1)
test_dataset = test_dataset.drop('MasVnrType', axis=1)
test_dataset = test_dataset.drop('GarageYrBlt', axis=1)
test_dataset = test_dataset.drop('Electrical', axis=1)


# taking care of missing data
C = test_dataset.isnull().sum()

test_dataset['BsmtQual'].value_counts(dropna=False)
test_dataset['BsmtQual'].fillna(value='TA', inplace=True)
test_dataset['BsmtCond'].value_counts(dropna=False)
test_dataset['BsmtCond'].fillna(value='TA', inplace=True)
test_dataset['BsmtExposure'].value_counts(dropna=False)
test_dataset['BsmtExposure'].fillna(value='No', inplace=True)
test_dataset['GarageType'].value_counts(dropna=False)
test_dataset['GarageType'].fillna(value='Attchd', inplace=True)
test_dataset['GarageFinish'].value_counts(dropna=False)
test_dataset['GarageFinish'].fillna(value='RFn', inplace=True)
test_dataset['GarageQual'].value_counts(dropna=False)
test_dataset['GarageQual'].fillna(value='TA', inplace=True)
test_dataset['GarageCond'].value_counts(dropna=False)
test_dataset['GarageCond'].fillna(value='TA', inplace=True)
test_dataset['BsmtFinType1'].value_counts(dropna=False)
test_dataset['BsmtFinType1'].fillna(value='ALQ', inplace=True)
test_dataset['BsmtFinType2'].value_counts(dropna=False)
test_dataset['BsmtFinType2'].fillna(value='Unf', inplace=True)
test_dataset['MasVnrArea'].value_counts(dropna=False)
test_dataset['MasVnrArea'].fillna(value=0.0, inplace=True)
test_dataset['MSZoning'].value_counts(dropna=False)
test_dataset['MSZoning'].fillna(value='RL', inplace=True)
test_dataset['Utilities'].value_counts(dropna=False)
test_dataset['Utilities'].fillna(value='AllPub', inplace=True)
test_dataset['Exterior1st'].value_counts(dropna=False)
test_dataset['Exterior1st'].fillna(value='VinylSd', inplace=True)
test_dataset['BsmtFinSF1'].value_counts(dropna=False)
test_dataset['BsmtFinSF1'].fillna(value=0.0, inplace=True)
test_dataset['BsmtFinSF2'].value_counts(dropna=False)
test_dataset['BsmtFinSF2'].fillna(value=0.0, inplace=True)
test_dataset['BsmtUnfSF'].value_counts(dropna=False)
test_dataset['BsmtUnfSF'].fillna(value=0.0, inplace=True)
test_dataset['TotalBsmtSF'].value_counts(dropna=False)
test_dataset['TotalBsmtSF'].fillna(value=0.0, inplace=True)
test_dataset['BsmtFullBath'].value_counts(dropna=False)
test_dataset['BsmtFullBath'].fillna(value=0.0, inplace=True)
test_dataset['BsmtHalfBath'].value_counts(dropna=False)
test_dataset['BsmtHalfBath'].fillna(value=0.0, inplace=True)
test_dataset['KitchenQual'].value_counts(dropna=False)
test_dataset['KitchenQual'].fillna(value='TA', inplace=True)
test_dataset['Functional'].value_counts(dropna=False)
test_dataset['Functional'].fillna(value='Typ', inplace=True)
test_dataset['GarageCars'].value_counts(dropna=False)
test_dataset['GarageCars'].fillna(value=2.0, inplace=True)
test_dataset['GarageArea'].value_counts(dropna=False)
test_dataset['GarageArea'].fillna(value=0.0, inplace=True)
test_dataset['SaleType'].value_counts(dropna=False)
test_dataset['SaleType'].fillna(value='WD', inplace=True)
test_dataset['Exterior2nd'].value_counts(dropna=False)
test_dataset['Exterior2nd'].fillna(value='VinylSd', inplace=True)


test_dataset['LotFrontage'].mean()
test_dataset['LotFrontage'].value_counts(dropna=False)
test_dataset['LotFrontage'].fillna(value=dataset['LotFrontage'].mean(), inplace=True)

# Encoding categorical data

B = test_dataset.iloc[:, 0:71].values


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
B[:, 1] = labelencoder.fit_transform(B[:, 1])
B[:, 4] = labelencoder.fit_transform(B[:, 4])
B[:, 5] = labelencoder.fit_transform(B[:, 5])
B[:, 6] = labelencoder.fit_transform(B[:, 6])
B[:, 7] = labelencoder.fit_transform(B[:, 7])
B[:, 8] = labelencoder.fit_transform(B[:, 8])
B[:, 9] = labelencoder.fit_transform(B[:, 9])
B[:, 10] = labelencoder.fit_transform(B[:, 10])
B[:, 11] = labelencoder.fit_transform(B[:, 11])
B[:, 12] = labelencoder.fit_transform(B[:, 12])
B[:, 13] = labelencoder.fit_transform(B[:, 13])
B[:, 14] = labelencoder.fit_transform(B[:, 14])
B[:, 19] = labelencoder.fit_transform(B[:, 19])
B[:, 20] = labelencoder.fit_transform(B[:, 20])
B[:, 21] = labelencoder.fit_transform(B[:, 21])
B[:, 22] = labelencoder.fit_transform(B[:, 22])
B[:, 24] = labelencoder.fit_transform(B[:, 24])
B[:, 25] = labelencoder.fit_transform(B[:, 25])
B[:, 26] = labelencoder.fit_transform(B[:, 26])
B[:, 27] = labelencoder.fit_transform(B[:, 27])
B[:, 28] = labelencoder.fit_transform(B[:, 28])
B[:, 29] = labelencoder.fit_transform(B[:, 29])
B[:, 30] = labelencoder.fit_transform(B[:, 30])
B[:, 32] = labelencoder.fit_transform(B[:, 32])
B[:, 36] = labelencoder.fit_transform(B[:, 36])
B[:, 37] = labelencoder.fit_transform(B[:, 37])
B[:, 38] = labelencoder.fit_transform(B[:, 38])
B[:, 49] = labelencoder.fit_transform(B[:, 49])
B[:, 51] = labelencoder.fit_transform(B[:, 51])
B[:, 54] = labelencoder.fit_transform(B[:, 54])
B[:, 53] = labelencoder.fit_transform(B[:, 53])
B[:, 58] = labelencoder.fit_transform(B[:, 58])
B[:, 59] = labelencoder.fit_transform(B[:, 59])
B[:, 57] = labelencoder.fit_transform(B[:, 57])
B[:, 70] = labelencoder.fit_transform(B[:, 70])
B[:, 69] = labelencoder.fit_transform(B[:, 69])








# Predicting the Test set results
y_pred = regressor.predict(B)

pd.DataFrame(y_pred).to_csv("Predictions.csv")










