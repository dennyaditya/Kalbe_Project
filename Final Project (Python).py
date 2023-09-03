#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pmdarima import auto_arima


# In[2]:


df_Customer = pd.read_csv (r"C:\Users\USER\Documents\course\Data Science\KALBE (Virtual Internship)\Minggu 4\Case Study - Customer.csv", delimiter = ";")
df_Product = pd.read_csv (r"C:\Users\USER\Documents\course\Data Science\KALBE (Virtual Internship)\Minggu 4\Case Study - Product.csv", delimiter = ";")
df_Store = pd.read_csv (r"C:\Users\USER\Documents\course\Data Science\KALBE (Virtual Internship)\Minggu 4\Case Study - Store.csv", delimiter = ";")
df_Transaction = pd.read_csv (r"C:\Users\USER\Documents\course\Data Science\KALBE (Virtual Internship)\Minggu 4\Case Study - Transaction.csv", delimiter = ";")


# In[3]:


df_Customer.head()


# In[4]:


df_Product.head()


# In[5]:


df_Store.head()


# In[6]:


df_Transaction.head()


# In[7]:


df_merge = pd.merge (df_Customer, df_Transaction, on =['CustomerID'])
df_merge = pd.merge (df_merge, df_Product.drop(columns= ['Price']), on=['ProductID'])
df_merge = pd.merge (df_merge, df_Store, on=['StoreID'])


# In[8]:


df_merge.head()


# In[9]:


df_merge.dtypes


# In[10]:


df_merge['Income'] = df_merge['Income'].replace('[,]', '.', regex=True).astype ('float')
df_merge['Latitude'] = df_merge['Latitude'].replace('[,]', '.', regex=True).astype('float')
df_merge['Longitude'] = df_merge['Longitude'].replace('[,]', '.', regex=True).astype('float')
df_merge['Date'] = pd.to_datetime(df_merge['Date'])


# In[11]:


df_merge.dtypes


# In[12]:


df_merge.shape


# In[13]:


df_merge.head()


# In[14]:


daily_qty = df_merge.groupby('Date')['Qty'].sum().reset_index()
daily_qty


# In[15]:


Train_size = int(0.9 * len(daily_qty))
df_train = daily_qty [:Train_size]
df_test = daily_qty [Train_size:]


# In[16]:


df_train


# In[17]:


df_test


# In[18]:


plt.figure(figsize=(20,5))
sns.lineplot(data=df_train, x=df_train["Date"], y=df_train["Qty"]);
sns.lineplot(data=df_test, x=df_test["Date"], y=df_test["Qty"]);


# In[19]:


df_train = df_train.set_index ('Date')
df_test = df_test.set_index ('Date')

y = df_train['Qty']


# In[20]:


plot_acf(df_train['Qty'])  
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('ACF Value')
plt.show()


# In[21]:


plot_pacf(df_train['Qty'])  
plt.title('Partial Autocorrelation Function (PACF)')
plt.xlabel('Lag')
plt.ylabel('PACF Value')
plt.show()


# In[22]:


decomposition = sm.tsa.seasonal_decompose(y, model='additive')


# In[23]:


plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(y, label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(decomposition.resid, label='Residual')
plt.legend(loc='upper left')
plt.tight_layout()

plt.show()


# In[24]:


adtest = adfuller(y)
adtest[1]


# In[25]:


ArimaModel = ARIMA(y, order=(10, 0, 9))
ArimaFit = ArimaModel.fit()


# In[26]:


ypred_steps = len(df_test)
ypred = ArimaFit.forecast(steps=ypred_steps)


# In[27]:


plt.figure(figsize=(12, 6))
plt.plot(y.index, y, label='Training Data')
plt.plot(df_test.index, df_test['Qty'], label='Test Data')
plt.plot(df_test.index, ypred, label='Forecast', color='black')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.title('ARIMA Forecast for Total Quantity')
plt.legend()
plt.show()


# In[28]:


rmse = np.sqrt(mean_squared_error(df_test, ypred))
print(f"RMSE: {rmse}")
mae = mean_absolute_error(df_test, ypred)
print(f"MAE: {mae}")


# In[29]:


df_cluster = df_merge.groupby(['CustomerID']).agg({
    'TransactionID': 'count',
    'Qty': 'sum',
    'TotalAmount': 'sum'
}).reset_index()
df_cluster


# In[30]:


data_cluster=df_cluster.drop(columns='CustomerID')
data_cluster


# In[31]:


df_cluster_normalize = preprocessing.normalize(data_cluster)
df_cluster_normalize


# In[32]:


K = range(2, 8)
fits = []
score = []
for k in K:
    model = KMeans(n_clusters = k, random_state = 0 , n_init='auto').fit(df_cluster_normalize)
    
    fits.append(model)
    
    score.append(silhouette_score(df_cluster_normalize, model.labels_, metric='euclidean'))


# In[33]:


sns.lineplot(x = K, y=score);


# In[34]:


df_cluster['cluster_label'] = fits[2].labels_


# In[35]:


df_cluster.groupby(['cluster_label']).agg({
    'CustomerID': [('Count', 'count')],
    'TransactionID': [('Mean', 'mean')],
    'Qty': [('Mean', 'mean')],
    'TotalAmount': [('Mean', 'mean')]
})

