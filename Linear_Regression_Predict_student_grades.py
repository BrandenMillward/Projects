# basic linear regression model to predict students grades

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

# source of the data set https://archive.ics.uci.edu/ml/datasets/Student+Performance
df = pd.read_csv("student-mat.csv", sep=";")
print(df.head())

# limit to the columns that I will be using
df = df[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(df)

# set what I am looking to predict
predict = "G3"

# separate the value or prediction and the values used to predict
x = np.array(df.drop(columns=predict))
y = np.array(df[predict])


# create a test and train split of 10:90
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# define and train the model
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

# assess the accuracy of the model
accuracy = linear.score(x_test, y_test)
print(accuracy)

# display the coefficients and intercept of the model
print("Co: \n", linear.coef_)
print("Intercept \n", linear.intercept_)

# check prediction against actuals
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
