#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
data=pd.read_csv("C:/Users/DELL/Desktop/IMDB Dataset.csv")
data


# In[15]:


from sklearn.naive_bayes import MultinomialNB


# In[16]:


data.isnull().sum()


# In[17]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['sentiment']=encoder.fit_transform(data.sentiment)


# In[18]:


data


# In[19]:


x=data.iloc[0:,0]
x


# In[20]:


y=data.iloc[0:,1]
y


# In[21]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.25, random_state=0)


# In[22]:


y_train


# In[23]:


from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer(stop_words='english')
vector.fit(x_train)


# In[24]:


x_train_transformed =vector.transform(x_train)
x_test_transformed =vector.transform(x_test)


# In[26]:


from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()

model.fit(x_train_transformed,y_train)
y_pred = model.predict(x_test_transformed)
y_pred_prob = model.predict_proba(x_test_transformed)
y_pred_prob


# In[27]:


from sklearn.metrics import confusion_matrix,accuracy_score
print(accuracy_score(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




