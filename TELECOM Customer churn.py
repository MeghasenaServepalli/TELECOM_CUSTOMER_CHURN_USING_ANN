#!/usr/bin/env python
# coding: utf-8

# In[2]:


#TELECOM Customer Churn Prediction using ANN
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[3]:


df=pd.read_csv("Telco-Customer-Churn.csv")


# In[4]:


df.sample()


# In[5]:


df.drop("customerID",axis='columns',inplace=True)


# In[6]:


df.dtypes


# In[7]:


df.TotalCharges.values


# In[8]:


df.MonthlyCharges.values


# In[9]:


pd.to_numeric(df.TotalCharges, errors='coerce').isnull()


# In[10]:


df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()].shape


# In[11]:


df.iloc[488]['TotalCharges']


# In[12]:


df1=df[df.TotalCharges!=' ']
df1.shape
df1.dtypes


# In[13]:


df1.TotalCharges=pd.to_numeric(df1.TotalCharges)
df1.TotalCharges.dtype


# In[14]:


df1.shape


# In[15]:


df1.dtypes


# In[16]:


df1.TotalCharges=df[pd.to_numeric(df1.TotalCharges)]


# In[17]:


df1.TotalCharges.dtypes


# In[18]:


df1.dtypes


# In[19]:


#histogram to check churn no of customers leaving and coming
#churn =no means customers not leaving
#tenure is the amount of time the cutsomers stay for a company
df1[df1.Churn=='No']


# In[20]:


tenure_churn_no=df1[df1.Churn=='No'].tenure
tenure_churn_no


# In[21]:


tenure_churn_yes=df1[df1.Churn=='Yes'].tenure
tenure_churn_yes


# In[22]:


tenure_churn_yes=df1[df1.Churn=='Yes'].tenure
tenure_churn_yes


# In[23]:


plt.xlabel("tenure")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualization")
plt.hist([tenure_churn_yes,tenure_churn_no],color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()


# In[24]:


tenure_churn_yes=df1[df1.Churn=='Yes'].MonthlyCharges
tenure_churn_no=df1[df1.Churn=='No'].MonthlyCharges
plt.xlabel("Monthly Charges")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualization")
plt.hist([tenure_churn_yes,tenure_churn_no],color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()


# In[25]:


#Label encoding- done on all the columns with yes or no fields and then find all the unique columns
def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes=='object':#get only object type columns
            print(f'{column} : {df[column].unique()}')
print_unique_col_values(df1)


# In[26]:


# data cleaning phase
df1.replace('No internet service','No',inplace=True)
df1.replace('No phone service','No',inplace=True)


# In[27]:


print_unique_col_values(df1)


# In[28]:


#convert all yes and no to 1 and 0 because Ml dont understand strings like yes and no
yes_no_columns=['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection',
               'TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    df1[col].replace({'Yes':1,'No':0},inplace=True)
    


# In[29]:


print_unique_col_values(df1)


# In[30]:


for col in df1:
    print(f'{col}:{df[col].unique()}')


# In[31]:


df1['gender'].replace({'Female':1,'Male':0},inplace=True)


# In[32]:


df1['gender'].unique()
df1.dtypes


# In[33]:


#Perform one-hot encoding(create all the columns separetely and gives values 1 and rest as 0s
#correspondigly)

df2=pd.get_dummies(data=df1,columns=['InternetService','Contract','PaymentMethod'])
df2.columns


# In[34]:


df2.head()


# In[35]:


df2.dtypes


# In[36]:


#Perform Scaling --convert to 0s and 1s from numerical values
# so here tenure, Monthly Charges, Total Charges
cols_to_scale=['tenure','MonthlyCharges','TotalCharges']
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

# after getting the scaler we can fit and transform in those columns
df2[cols_to_scale]=scaler.fit_transform(df2[cols_to_scale])


# In[37]:


df2.sample(3)


# In[38]:


for col in df2:
    print(f'{col}:{df2[col].unique()}')


# In[50]:


# train and test split get X and Y axis
X=df2.drop('Churn',axis='columns')
y=df2['Churn']


# In[51]:


from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)


# In[52]:


X_train.shape


# In[53]:


X_test.shape


# In[54]:


X_train[:10]


# In[55]:


len(X_train.columns)


# In[56]:


pip install tensorflow


# In[57]:


pip install keras


# In[58]:


pip show tensorflow


# In[59]:


pip show keras


# In[65]:


#import TensorFlow
import tensorflow as tf
from tensorflow import keras
# lets create a neural network usually there is 1 input and 1 output layer
from keras.layers import Input, Dense
model=keras.Sequential([
    keras.layers.Dense(20,input_shape=(26,),activation='relu'),
   # keras.layers.Dense(15,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid'),
])
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])
model.fit(X_train,y_train,epochs=50)


# In[66]:


model.evaluate(X_test,y_test)


# In[72]:


yp=model.predict(X_test)
yp[:10]


# In[69]:


y_test[:5]


# In[70]:


y_pred=[]
for ele in yp:
    if ele > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)


# In[73]:


y_pred[:10]


# In[74]:


#get classification report
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test,y_pred))


# In[76]:


#get the confusion matrix
import seaborn as sn
cm=tf.math.confusion_matrix(labels=y_test,predictions=y_pred)
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[78]:


#accuracy and then compare both cm and cr
round((887+221)/(887+112+187+221),2)


# In[79]:


#precision for 0th class i.e; precision for customers who did not churn
round((887)/(887+187),2)


# In[80]:


#Precison for 1 class ie; Precision for customers who did churn
round(221/(221+112),2)

