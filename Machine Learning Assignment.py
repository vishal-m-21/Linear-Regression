#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


dataset=pd.read_csv('LinearRegression_EntriesGender.csv')


# In[5]:


dataset


# In[6]:


dataset.head()


# In[7]:


dataset.info()


# In[8]:


dataset.isnull().sum()


# In[9]:


from sklearn import linear_model
X = dataset['Female'].values
Y = dataset['Male'].values


# In[10]:


#mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

#total number of values 
m = len(X)

#using the formula to calculate b1 and b0
numer = 0
denom = 0
for i in range(m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

#print coefficients
print(b1,b0)


# In[11]:


#To_find_the accuracy
ss_t = 0
ss_r = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) **2
    ss_r += (Y[i] - y_pred) **2
    
r2 = 1 - (ss_r/ss_t)
print(r2)


# In[12]:


#plotting  values and regression line

max_x = np.max(X) + 100
min_x = np.min(X) - 100

#calulating line values x and y
x = np.linspace(min_x,max_x, 1000)
y = b0+b1 * x

#ploting line
plt.plot(x,y,color="#58b970", label='Regression Line')
#ploting Scatter points
plt.scatter(X,Y, c ='#ef5423', label ='Scatter Plot')

plt.xlabel('Female')
plt.ylabel('Male')
plt.legend()
plt.show()


# In[ ]:




