#!/usr/bin/env python
# coding: utf-8

# # Linear Regression with Python Scikit Learn
# 
# Requirements: Develop a linear regression model predicting the percentage score of a student based on the number of hours they study. This is a simple simple linear regression task that includes only two variables: hours of study and percentage score.
# 
# ## Simple Linear Regression
# 

# In[4]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# Reading data from remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)


# We will first plot our data points on a 2-D graph to observe our dataset to see if there is any relationship between the data

# In[11]:


# Data Visualization
sns.set(style="darkgrid")
s_data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# ## Preparing the data
# We will now divide the data into attributes and labels

# In[12]:


X = s_data.iloc[:,:-1].values
y = s_data.iloc[:,1].values


# Next, we will split the data into training and testing sets

# The graph above shows that there is a **positive linear relation** between the number of hours studied and the percentage of score. 

# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ## Training the model

# In[15]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[16]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_ # According to the linear equation

# Plotting for the test data
plt.scatter(X,y)
plt.plot(X, line);
plt.show()


# ## Making Predictions

# In[20]:


print(X_test)
y_pred = regressor.predict(X_test) #Predicting the scores

# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# In[22]:


# Test with chosen input, let's say we want to predict the score of 9.25 hours studied
hours = 9.25

own_pred = regressor.predict([[hours]])

print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# ## Evaluating the model

# In[26]:


from sklearn import metrics
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)


# MAE of roughly 4.18 means that on average, the difference between the actual scores and the predicted scores is about 4.18 percentage points
