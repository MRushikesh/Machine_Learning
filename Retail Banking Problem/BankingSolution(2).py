#!/usr/bin/env python
# coding: utf-8

# # <center> Banking Solution</center>

# In[1]:


#importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier


# In[2]:


#getting dataset
dataset = pd.read_csv("C:/Users/kabira/Desktop/banking.csv")


# In[3]:


dataset


# In[4]:


dataset["pdays"].value_counts()


# In[5]:


for i in range(41188):    
    if (dataset.loc[i]['pdays'] ==999):
        dataset.loc[i,'pdays'] = 0


# In[6]:


dataset["pdays"].value_counts()


# In[7]:


#checking for Nan Values
dataset.isnull().sum()


# In[8]:


dataset.head()


# In[9]:


dataset.corr()


# In[10]:


dataset.describe(include="object")


# In[12]:


df = dataset[['age','emp_var_rate','cons_price_idx','cons_conf_idx','euribor3m']]

sns.pairplot(df, kind="scatter")
plt.show()


# In[13]:


#taking relation between job and education so we can fill some of the unknown values
pd.crosstab(dataset.job, dataset.education)


# In[14]:


dataset['education'].value_counts()


# In[15]:


dataset['job'].value_counts()


# In[16]:


dataset.shape


# In[17]:


for i in range(41188):    
    if (dataset.loc[i]['age']>60) & (dataset.loc[i]['job']=='unknown'):
        dataset.loc[i,'job'] ='retired'


# In[18]:


for i in range(41188):
    if (dataset.loc[i]['job']=='management') & (dataset.loc[i]['education'] == 'unknown'):
        dataset.loc[i,'education'] = 'university.degree'


# In[19]:


for i in range(41188):
    if (dataset.loc[i]['job']=='services') & (dataset.loc[i]['education'] == 'unknown'):
        dataset.loc[i ,'education'] = 'high.school'


# In[20]:


for i in range(41188):
    if (dataset.loc[i]['job']=='household') & (dataset.loc[i]['education'] == 'unknown'):
        dataset.loc[i, 'education'] = 'basic.4y'


# In[21]:


for i in range(41188):
    if (dataset.loc[i]['job']=='unknown') & (dataset.loc[i]['education'] == 'basic.4y'):
        dataset.loc[i, 'job'] = 'blue-collar'


# In[22]:


for i in range(41188):
    if (dataset.loc[i]['job']=='unknown') & (dataset.loc[i]['education'] == 'basic.6y'):
        dataset.loc[i, 'job'] = 'blue-collar'


# In[23]:


for i in range(41188):
    if(dataset.loc[i]['job']=='unknown') & (dataset.loc[i]['education'] == 'basic.9y'):
        dataset.loc[i, 'job'] = 'blue-collar'


# In[25]:


dataset['education'].value_counts()


# In[27]:


dataset['job'].value_counts()


# In[26]:


data = dataset.iloc[:, :-1].values


# In[27]:


result = dataset.iloc[:, 20].values


# In[28]:


result.shape


# In[29]:


data.shape


# In[30]:


data


# In[70]:


result


# In[32]:


#label encoding the values
labelencoder_data = LabelEncoder()
data[:, 1] = labelencoder_data.fit_transform(data[:, 1])


# In[33]:


data[:, 2] = labelencoder_data.fit_transform(data[:, 2])
data[:, 3] = labelencoder_data.fit_transform(data[:, 3])
data[:, 4] = labelencoder_data.fit_transform(data[:, 4])
data[:, 5] = labelencoder_data.fit_transform(data[:, 5])
data[:, 6] = labelencoder_data.fit_transform(data[:, 6])
data[:, 7] = labelencoder_data.fit_transform(data[:, 7])
data[:, 8] = labelencoder_data.fit_transform(data[:, 8])
data[:, 9] = labelencoder_data.fit_transform(data[:, 9])
data[:, 14] = labelencoder_data.fit_transform(data[:, 14])


# In[35]:


data


# In[36]:


#tran_test split
from sklearn.model_selection import train_test_split
data_train, data_test, result_train, result_test = train_test_split(data, result, test_size = 0.2, random_state = 0)


# In[42]:


print(data_train.shape)
print(data_test.shape)
print(result_train.shape)
print(result_test.shape)


# In[45]:


# applying AdaBoost Classifier Method
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# In[46]:


classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),n_estimators=50)
classifier.fit(data_train,result_train)


# In[47]:


classifier.score(data_train,result_train)


# In[48]:


classifier.score(data_test,result_test)


# In[49]:


data_pred = classifier.predict(data_test)


# In[50]:


data_pred


# In[51]:


result_test


# In[52]:


cm = confusion_matrix(result_test, data_pred)


# In[53]:


cm


# In[56]:


# Appying Logistic Regression Method
from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression(random_state = 0)
classifier2.fit(data_train, result_train)


# In[57]:


classifier2.score(data_train, result_train)


# In[58]:


classifier2.score(data_test, result_test)


# In[59]:


data_pred2 = classifier.predict(data_test)


# In[60]:


cm2 =confusion_matrix(result_test, data_pred2)


# In[61]:


cm2


# In[71]:


#using XGboost Classifier
from xgboost import XGBClassifier
classifier3 = XGBClassifier()
classifier3.fit(data_train, result_train)


# In[72]:


classifier3.score(data_train, result_train)


# In[74]:


classifier3.score(data_test, result_test)


# In[76]:


#Checking for accuracy using confusion matrix
data_pred3 = classifier.predict(data_test)
cm3 =confusion_matrix(result_test, data_pred3)
cm3


# In[ ]:




