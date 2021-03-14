#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[10]:


data = pd.read_csv("housing.csv", delimiter=r"\s+", header=None)
data.head()


# In[13]:


import seaborn as sns
corr_matrix = data.corr()
plt.figure(figsize=(10,5))
sns.heatmap(corr_matrix, annot=True)


# i am plotting corelation matrix of data and i found that the 5th column i.e, 
# rm average number of rooms per dwelling has a strong corelation of 0.7 with the 
# 13th column that is medv median value of owner-occupied homes in $1000s. So i am picking 5th column to predict 13th column

# In[38]:


X = data.iloc[:,5]
y = data.iloc[:,13]


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[21]:


reg = LinearRegression()
reg.fit(x_train.values.reshape(-1,1),y_train)


# In[26]:


from sklearn.metrics import mean_squared_error, r2_score

y_pred = reg.predict(x_test.values.reshape(-1,1))
rmse_Score = mean_squared_error(y_test,y_pred, squared=False)
r2_Score = r2_score(y_test,y_pred)

print("rmse score of the data: ", rmse_Score)
print("r2 score of the data: ", r2_Score)


# In[52]:


y_predicted = reg.predict(X.values.reshape(-1,1))
plt.scatter(X,y)
plt.plot(X,y_predicted)
plt.title("Best fit line from the linear regression")
plt.xlabel("number of rooms (x_test)")
plt.ylabel("median price (y_test)")
plt.show()


# In[122]:


from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X.values.reshape(-1,1))


# In[123]:


X_poly.shape


# In[124]:


x_train, x_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=1)


# In[125]:


reg_poly = LinearRegression()
reg_poly.fit(x_train,y_train)


# In[126]:


y_pred = reg_poly.predict(x_test)
rmse_Score_poly = mean_squared_error(y_test,y_pred, squared=False)
r2_Score_poly = r2_score(y_test,y_pred)

print("rmse score of the data: ", rmse_Score_poly)
print("r2 score of the data: ", r2_Score_poly)


# In[127]:



y_predicted = reg_poly.predict(X_poly)
x_space = np.arange(3, 9, 1)
plt.scatter(x_space,y[0:6])
plt.plot(x_space,y_predicted[0:6])
plt.title("Best fit line from the linear regression")
plt.xlabel("number of rooms (x_test)")
plt.ylabel("median price (y_test)")
plt.show()


# In[128]:


poly_20 = PolynomialFeatures(degree=20)
X_poly_20 = poly.fit_transform(X.values.reshape(-1,1))


# In[129]:


x_train, x_test, y_train, y_test = train_test_split(X_poly_20, y, test_size=0.2, random_state=1)


# In[130]:


reg_poly_20 = LinearRegression()
reg_poly_20.fit(x_train,y_train)


# In[131]:


y_pred = reg_poly_20.predict(x_test)
rmse_Score_poly_20 = mean_squared_error(y_test,y_pred, squared=False)
r2_Score_poly_20 = r2_score(y_test,y_pred)

print("rmse score of the data: ", rmse_Score_poly_20)
print("r2 score of the data: ", r2_Score_poly_20)


# In[132]:



y_predicted = reg_poly_20.predict(X_poly_20)
x_space = np.arange(3, 9, 1)
plt.scatter(x_space,y[0:6])
plt.plot(x_space,y_predicted[0:6])
plt.title("Best fit line from the linear regression")
plt.xlabel("number of rooms (x_test)")
plt.ylabel("median price (y_test)")
plt.show()


# In[112]:


x_multiple = data.iloc[:,[5,11,12]]
y = data.iloc[:,13]


# In[114]:


x_train, x_test, y_train, y_test = train_test_split(x_multiple, y, test_size=0.2, random_state=1)
reg_multiple = LinearRegression()
reg_multiple.fit(x_train,y_train)


# In[117]:


y_pred = reg_multiple.predict(x_test)
rmse_Score_multiple = mean_squared_error(y_test,y_pred, squared=False)
r2_Score_multiple = r2_score(y_test,y_pred)

print("rmse score of the data: ", rmse_Score_multiple)
print("r2 score of the data: ", r2_Score_multiple)

