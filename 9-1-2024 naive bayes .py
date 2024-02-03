#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes Classifier in Python

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


dataset=pd.read_csv(r"D:\Data Science with AI\8th-jan-2023\project\adult.csv")


# In[4]:


dataset


# In[5]:


dataset.shape


# In[6]:


dataset.head()


# In[7]:


#rename columns names

col_names=['age' ,'workclass','fnlwgt',	'education',	'education.num',	'marital.status' ,	'occupation','relationship','race','sex','capital.gain','capital.loss' ,'hours.per.week','native.country','income']
dataset.columns=col_names
dataset.columns


# In[8]:


# let's again preview the dataset

dataset.head()


# In[9]:


#summary
dataset.info()


# In[10]:


#explore categorical variable

categorical=[var for var in dataset.columns if dataset[var].dtype=='O']

print('there are {} categorical variables\n'.format(len(categorical)))

print('the categorical variables are :\n\n', categorical)


# In[11]:


# view the categorical variables

dataset[categorical].head()


# In[12]:


#missing value in categorical values 

dataset[categorical].isnull().sum()


# In[13]:


# frequency count of values in categorical variables 


# In[14]:


for var in categorical:
    print(dataset[var].value_counts())


# In[15]:


#view frequecy distribution of categorical variables 

for var in categorical:
    print(dataset[var].value_counts()/np.float16(len(dataset)))


# In[ ]:





# In[16]:


#exploare workclass variable

dataset.workclass.unique()


# In[17]:


# replace '?' values in workclass variable with 'NAN'

dataset['workclass'].replace('?',np.NAN, inplace=True)


# In[18]:


# again check the frequency distribution of values in workclass variable
dataset.workclass.value_counts()


# In[19]:


# explore occupation variable
dataset.occupation.unique()


# In[20]:


# check frequency distribution of values in occupation variable
dataset.occupation.value_counts()


# In[21]:


# replace'?' values in occupation variable with nan 
dataset['occupation'].replace('?',np.NAN, inplace=True)


# In[22]:


# check the frequency distribution values in occupation variable

dataset.occupation.value_counts()


# # explore native_country variable
# 
# 

# In[23]:


#check labels in native country variable

dataset['native.country'].unique()


# In[24]:


dataset


# In[25]:


# check frequency distribution of values in native_country variable
dataset['native.country'].value_counts()


# In[26]:


# replace '?' values in native_country variable with 'NAN'

dataset['native.country'].replace('?',np.NAN,inplace=True)


# In[27]:


#check frequency distribution values

dataset['native.country'].value_counts()


# # check missing values in categorical variables again

# In[28]:


dataset[categorical].isnull().sum()


# # number of labels: cardinality

# In[29]:


# check for cardinaliity in categorical variable 
for var in categorical:
    print(var,'contains',len(dataset[var].unique()),'labels')


# # explore numerical variables

# In[30]:


# find numerical variable

numerical=[var for var in dataset.columns if dataset[var].dtype!='o']
print('there are {} numerical variables\n'.format(len(numerical)))
print('the numeric variables are :',numerical)


# In[31]:


# view numerical variables

dataset[numerical].head()


# # missing values in numerical variable

# In[32]:


# check missing values in numerical variables 

dataset[numerical].isnull().sum()


# # declare feature vector and target variable

# In[33]:


x=dataset.drop(['income'],axis=1)
y=dataset['income']


# # split the data into training and testing set

# In[34]:


# split x and y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[35]:


# check the shape of x_train and x_test

x_train.shape,x_test.shape


# # featuere engineering

# In[36]:


# check datatype in x_train
x_train.dtypes


# In[37]:


# display categorical variables
categorical=[col for col in x_train.columns if x_train[col].dtypes=='O']
categorical


# In[38]:


# display numerical variables

numerical=[col for col in x_train.columns if x_train[col].dtypes!='O']
numerical


# # engineering missing values in categorical variables

# In[39]:


# print percentage of missing values in categorical variables

x_train[categorical].isnull().mean()


# In[40]:


#print categorical variables with missing data

for col in categorical:
    if x_train[col].isnull().mean()>0:
        print(col, (x_train[col].isnull().mean()))


# In[41]:


#impute missing categorical variables with most frequent value 

for df2 in [x_train,x_test]:
    df2['workclass'].fillna(x_train['workclass'].mode()[0],inplace=True)
    df2['occupation'].fillna(x_train['occupation'].mode()[0],inplace=True)
    df2['native.country'].fillna(x_train['native.country'].mode()[0],inplace=True)
    


# In[42]:


#check missing values in categorical variables in x_train

x_train[categorical].isnull().sum()


# In[43]:


x_test[categorical].isnull().sum()


# In[44]:


#checking missing values in x_train
x_train.isnull().sum()


# In[45]:


x_test.isnull().sum()


# # encode categorical variable 

# In[46]:


#print categorical variable
categorical


# In[47]:


x_train[categorical].head()


# In[48]:


#!pip install --upgrade category_encoders


# In[49]:


# import categorical encoder

import category_encoders as ce


# In[50]:


x_train.head()


# In[51]:


x_train.shape


# In[52]:


x_test.head()


# In[53]:


x_train.shape


# In[54]:


x_train.head()


# # feature scaling

# In[55]:


cols=x_train.columns


# In[56]:


x_train


# In[57]:


#converting categorical to numeriacl data 


# In[58]:


from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
x_train1=scaler.fit_transform(x_train[['age']])
x_train


# In[60]:


x_train['workclass']


# In[61]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
x_train=lb.fit_transform(x_train[['workclass']])


# In[62]:


x_train


# In[ ]:





# In[64]:


from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
x_train2=scaler.fit_transform(x_train.reshape(-1,1))
x_train2


# In[ ]:


x_train


# In[ ]:





# In[ ]:


x_train = pd.DataFrame(x_train,columns=[cols])


# In[ ]:


x_test=pd.DataFrame(x_test,columns=[cols])


# In[ ]:


x_train.head()


# # model training

# In[ ]:


#train a gaussian naive bayes classifier on the training set
from sklearn.Naive_bayes import GaussianNB

#instantiate the model 
gnb=GaussianNB()

#fit the model
gnb.fit(x_train,y_train)


# # predict the result

# In[ ]:


y_pred=gnb.predict(x_test)
y_pred


# # check accuracy score 
# 

# In[ ]:


from sklearn.metrics import accuracy_score
print('model accuracy score:{0:0.4f}'.format(accuracy_score(y_test,y_pred)))


# ##### here y_test is the true class labels and y_pred is the predicted class labes in the test set

# # compare the train set and test set accuracy

# ###### now we will compare the train set and test set accuracy to check for overfitting

# In[ ]:


y_pred_train=gnb.predict(x_train)

y_pred_train


# In[ ]:


print('Training-set  accuracy score:{0:0.4f}'.format(accuracy_score(y_train,y_pred_train)))


# # check for overfitting and underfitting

# In[ ]:


#print the scores on training and test set

print('training set score :{:.4f}'.format(gnb.score(x_train,y_train)))
print('test set score :{:.4f}'.format(gnb.score(x_test,y_test)))


# # compare model accuracy with null accuracy 

# In[ ]:


#check class distribution in test set
y_test.value_counts()


# In[ ]:


#check null accuracy score 
null_accuracy =(7407/(7407+2362))

