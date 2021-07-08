#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


import sklearn
print(sklearn.__version__)


# In[12]:


# Here we are using inbuilt dataset of scikit learn
from sklearn.datasets import load_breast_cancer

# instantiating
cancer = load_breast_cancer()


# In[13]:


cancer.keys()


# In[17]:


print(cancer['feature_names'])


# In[26]:


# creating dataframe
df = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])

# checking head of dataframe
df.head()


# In[28]:


df.shape


# In[29]:


# Importing standardscalar module
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

# fitting
scalar.fit(df)
scaled_data = scalar.transform(df)

# Importing PCA
from sklearn.decomposition import PCA

# Let's say, components = 2
pca = PCA(n_components = 2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

x_pca.shape


# In[30]:


x_pca


# In[9]:


pca.fit(scaled_data)


# In[4]:


# giving a larger plot
plt.figure(figsize =(8, 6))

plt.scatter(x_pca[:, 0], x_pca[:, 1], c = cancer['target'], cmap ='plasma')

# labeling x and y axes
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')


# In[31]:


# components
pca.components_


# In[32]:


df_comp = pd.DataFrame(pca.components_, columns = cancer['feature_names'])

plt.figure(figsize =(14, 6))

# plotting heatmap
sns.heatmap(df_comp)


# In[ ]:




