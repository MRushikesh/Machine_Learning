#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# In[2]:


#getting dataset
dataset = pd.read_csv("C:/Users/kabira/Desktop/Upvotes Analysis/train.csv")


# In[3]:


dataset.head()


# In[4]:


dataset = dataset.drop("ID" , axis = 1)


# In[5]:


dataset = dataset.drop("Username" , axis = 1)


# In[6]:


dataset.head()


# In[7]:


dataset.shape


# In[8]:


dataset.isnull().sum()


# In[9]:


dataset.describe()


# In[10]:


plt.figure(figsize = (10,6) )
plot = dataset.corr()
sns.heatmap(plot,annot = True)


# In[11]:


from sklearn.preprocessing import Binarizer
bn = Binarizer(threshold =4)
new = bn.transform([dataset['Answers']])[0]
dataset['new'] = new
print(type(dataset))
dataset.head()


# In[12]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
dataset['Tag'] = lb.fit_transform(dataset['Tag'])
dataset.head()


# In[13]:


dataset.shape


# In[14]:


y = dataset.iloc[:, 4].values


# In[15]:


dataset = dataset.drop("Upvotes",axis = 1)


# In[16]:


dataset.head()


# In[17]:


X = dataset.iloc[:, 0:5].values


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 200)


# In[19]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[20]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 240)
regressor.fit(X_train, y_train)


# In[21]:


regressor.score(X_train, y_train)


# In[22]:


regressor.score(X_test, y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[23]:


testset = pd.read_csv("C:/Users/kabira/Desktop/Upvotes Analysis/test.csv")


# In[24]:


ID = testset['ID']


# In[25]:


testset = testset.drop("ID" , axis = 1)


# In[26]:


testset = testset.drop("Username" , axis = 1)


# In[27]:


testset.isnull().sum()


# In[28]:


testset.head()


# In[29]:


testset.shape


# In[30]:


le_test = LabelEncoder()
testset['Tag'] = le_test.fit_transform(testset['Tag'])
testset.head()


# In[31]:


from sklearn.preprocessing import Binarizer
bn = Binarizer(threshold =4)
new = bn.transform([testset['Answers']])[0]
testset['new'] = new
print(type(testset))
testset.head()


# In[32]:


testset['Tag'] = lb.fit_transform(testset['Tag'])


# In[33]:


testset.head()


# In[34]:


Z = testset.iloc[:, 0:5].values


# In[35]:


Z = sc.fit_transform(Z)


# In[36]:


pred = regressor.predict(Z)


# In[37]:


pred = abs(pred)


# In[40]:


sub = pd.DataFrame({'ID':ID,'Upvotes':pred})
sub.to_csv('C:/Users/kabira/Desktop/Upvotes Analysis/predictionsDelta.csv', index = False)


# In[ ]:





# In[ ]:




