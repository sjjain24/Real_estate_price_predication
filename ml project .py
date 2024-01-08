#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams["figure.figsize"]=(20,10)


# In[2]:


df=pd.read_csv("loan.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df['LoanAmount_log']=np.log(df['LoanAmount'])


# In[7]:


df['LoanAmount_log'].hist(bins=20);


# In[8]:


df['totalAmount']=df['ApplicantIncome']+df['CoapplicantIncome']
df['totalAmount_log']=np.log(df['totalAmount'])
df['totalAmount_log'].hist(bins=20)


# In[9]:


df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)   
df['Married'].fillna(df['Married'].mode()[0],inplace=True)   
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)   
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True) 

df.LoanAmount=df.LoanAmount.fillna(df.LoanAmount.mean())   
df.LoanAmount_log=df.LoanAmount_log.fillna(df.LoanAmount_log.mean())  

df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)   


# In[10]:


df.isnull().sum()


# In[11]:


print("no. of people who take loan as group by gender:")
print(df['Gender'].value_counts())
sns.countplot(x='Gender',data =df, palette='Set2');


# In[12]:


print("no. of people who take loan as group by martial status:")
print(df['Married'].value_counts())
sns.countplot(x='Married',data =df, palette='Set2');


# In[13]:


print("no. of people who take loan as group by Dependents:")
print(df['Dependents'].value_counts())
sns.countplot(x='Dependents',data =df, palette='Set2');


# In[14]:


print("no. of people who take loan as group by Self_Employed:")
print(df['Self_Employed'].value_counts())
sns.countplot(x='Self_Employed',data =df, palette='Set2');


# In[15]:


print("no. of people who take loan as group by LoanAmount :")
print(df['LoanAmount'].value_counts())
sns.countplot(x='LoanAmount',data =df, palette='Set2');


# In[16]:


print("no. of people who take loan as group by Credit_History :")
print(df['Credit_History'].value_counts())
sns.countplot(x='Credit_History',data =df, palette='Set2');


# In[17]:


df.head(10)


# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import LabelEncoder
label_encoder_X = LabelEncoder()

# Load data
df = pd.read_csv('loan.csv')



# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Preprocessing and feature selection
df = df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Loan_Status']]
df.dropna(inplace=True)


# In[4]:


X = df.drop('Loan_Status', axis='columns')
X.head()


# In[5]:


y = df['Loan_Status']
y.head()


# In[6]:


# Split into training and test sets

# Create logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print('Accuracy:', accuracy_score(y_test, y_pred))


# In[39]:


for i in range(0,5):

    X_train[:,i]=label_encoder.fit_transform(X_train[:,i])
    
    X_train[:,7]=label_encoder.fit_transform(X_train[:,7])
X_train


# In[40]:


for i in range(len(X_train.columns)):
    if i != 7:
        X_train[:, i] = label_encoder.fit_transform(X_train[:, i])


# In[ ]:




