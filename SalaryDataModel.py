#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import the necessary dependancies
import pandas as pd
import numpy as np
import matplotlib as plt

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv("SalaryData.csv")


# In[3]:


# Understand a little bit more about the data
df.head()


# In[4]:


# See how many data points we have
df


# In[5]:


# Understand the null values and in what categories they are in
df.info()


# In[6]:


# Since there aren't too many we will drop the na datapoints, kaggle and this dataset in particular
# is very easy to work with and well kept
df = df.dropna()


# In[7]:


# Understand the null values and in what categories they are in
df.info()


# In[8]:


df['Salary'].unique()


# In[12]:


df["Job Title"].unique()


# In[13]:


df["Job Title"].nunique()
# I don't know if we want to include job title, but we can do a sort of hot encoding
# I can ask to see how to make a duplicate of a column and parse the data to represent different industries
# plus an indication if they have a superlative with that title.

# Rank: Senior, Junior, Director, Manager, Specialtists
# Industries: Tech Support, Tech, Sales, Engineering, Finance

# maybe later tho


# In[ ]:


df['Salary'].unique()


# In[14]:


# Drop the outlier value
df = df.drop(df[df['Salary'] == 350].index)


# In[9]:


df['Salary'].unique()


# In[10]:


# This is the format we want to see all of our categories in, a number based implementation
df['Age'].unique()


# In[11]:


# Because we are dealing with a neural network, we are going to need to change the values of our qualitative data
# to a number system to be read by our model
df["Gender"].unique()


# In[12]:


# Since there are two different genders within this dataset, we can say that 0 is for male and 1 is for female
df['Gender'].replace('Male', 0, inplace=True)
df['Gender'].replace('Female', 1, inplace=True)


# In[13]:


df["Gender"].unique()


# In[14]:


# Now we can do the same with educaiton level
df["Education Level"].unique()


# In[15]:


# We will make our key as 0: Bachelor's degree, 1: Master's degree, 2: PhD
df['Education Level'].replace("Bachelor's", 0, inplace=True)
df['Education Level'].replace("Master's", 1, inplace=True)
df['Education Level'].replace("PhD", 2, inplace=True)


# In[16]:


df["Education Level"].unique()


# In[17]:


# Now we are going to get rid of Job title
# Although we believe that it may be important to consider due to the fact that there are senior positions,
# It is harder to classify in numerical data
df = df.drop(['Job Title'], axis=1)


# In[18]:


# View the dataset that we have now
df.head()


# In[15]:


# Create the dependent and independent variables
X = df.drop(["Salary"], axis=1)
y = df.drop(["Age", "Gender", "Education Level", "Years of Experience"], axis=1)


# In[20]:


y


# In[21]:


# Split the data into train and test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)


# In[33]:


# Standardizing the features
scaler = StandardScaler()
XtrainScaled = scaler.fit_transform(Xtrain)
XtestScaled = scaler.transform(Xtest)


# In[34]:


ytrain


# In[35]:


Xtrain


# In[36]:


# make sure that there aren't any nan values within our model
np.isnan(Xtrain).any()


# In[37]:


# checking the data type of certain columns
df['Years of Experience'].dtype


# In[38]:


trainds = Xtrain.join(ytrain)


# In[39]:


trainds.hist()


# In[40]:


# Neural Network Architecture
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(XtrainScaled.shape[1],)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)  # No activation for output layer (linear activation by default)
])


# In[41]:


# Compile the model
## NEED TO ADD WHAT ALL OF THESE ARE + EPLAINATIONS
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# In[42]:


# Create a new variable to fit/train the model with the training data
history = model.fit(
    XtrainScaled, 
    ytrain, 
    epochs=50, 
    batch_size=8, 
    validation_data=(XtestScaled, ytest)
)


# In[43]:


mse, mae = model.evaluate(XtestScaled, ytest)
print(f"Mean Squared Error on test set: {mse}")
print(f"Mean Absolute Error on test set: {mae}")


# In[44]:


predictions = model.predict(XtestScaled)


# In[45]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
predictions_actual = scaler.inverse_transform(np.concatenate([XtestScaled, predictions], axis=1))[:, -1]


# In[46]:


print(predictions)


# In[47]:


ytest


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




