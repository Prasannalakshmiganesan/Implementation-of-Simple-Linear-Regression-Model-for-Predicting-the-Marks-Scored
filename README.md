# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1 Import necessary libraries (e.g., pandas, numpy,matplotlib).
2 Load the dataset and then split the dataset into training and testing sets using sklearn library.
3 Create a Linear Regression model and train the model using the training data (study hours as input, marks scored as output).
4 Use the trained model to predict marks based on study hours in the test dataset.
5 Plot the regression line on a scatter plot to visualize the relationship between study hours and marks scored.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Prasannalakshmi G
RegisterNumber: 212222240075
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/MLSET.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='orange')
lr.coef_
lr.intercept_
```
## Output:

## 1)HEAD:
![image](https://github.com/Prasannalakshmiganesan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118610231/7f5c51da-9ec1-4d03-aae2-b282dba7bbaf)

## 2)GRAPH OF PLOTTED DATA:
![image](https://github.com/Prasannalakshmiganesan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118610231/f772be72-ccb5-4022-9dbc-bea28cf8e7cf)

## 3)TRAINED DATA:
![image](https://github.com/Prasannalakshmiganesan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118610231/3ccc9389-0fac-438e-abb1-ec20eeb937c7)

## 4)LINE OF REGRESSION:
![image](https://github.com/Prasannalakshmiganesan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118610231/1611acb7-a808-49b8-b811-a424444517d6)

## 5)COEFFICIENT AND INTERCEPT VALUES:
![image](https://github.com/Prasannalakshmiganesan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118610231/35d269e0-e13d-4afd-8b08-12060fbaa145)

```


##Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
