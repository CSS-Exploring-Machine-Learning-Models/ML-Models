import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Import data
df = pd.read_csv('Salary Data.csv')
df.info()

#Drop duplicates and missing values
df.dropna(inplace=True)
df.drop_duplicates()
df.info()

#Hot encode gender (female 0, male 1)
gender_label = LabelEncoder()
df['Gender']=gender_label.fit_transform(df['Gender'])

#Hot encode education level (0=bachelors, 1=masters, 2=phd)
edu_label_encoder = LabelEncoder()
df['Education Level'] = edu_label_encoder.fit_transform(df['Education Level'])

#Hot encode job title (174 unique values)
job_title_encoder = LabelEncoder()
df['Job Title']=job_title_encoder.fit_transform(df['Job Title'])

#Set target variable and drop it from the data frame
Y = df['Salary']
X = df.drop(['Salary'], axis=1)

#Splits the dataset into test/training sets, and uses the 20% of the data, and a seed of 69
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=69)

#Sets depth to 5, and trains the model
model=DecisionTreeRegressor(max_depth=5)
model.fit(x_train, y_train)
print('Model Score : ', model.score(x_test, y_test)*100)
y_pred = model.predict(x_test)
print('r2 : ', metrics.r2_score(y_pred, y_test)*100)


#Prints out actual and predicted data points, and the difference between them
'''
for i in range(len(y_test)):
    predicted_value = y_pred[i]
    actual_value = y_test.values[i]
    difference = predicted_value - actual_value
    
    print(f"Actual: {actual_value}, Predicted: {predicted_value}, Difference: {difference}")
'''

#Prints plot with x=actual salary, y=predicted salary, and a line representing a perfect prediction
'''
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Actual vs Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3, color='red')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
'''



