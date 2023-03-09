import ray
import pandas as pd
import numpy as np

from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# https://www.simplilearn.com/tutorials/scikit-learn-tutorial/sklearn-linear-regression-with-examples


# read data
df = pd.read_csv("/root/SYS-304/wo_men.csv")
print(df.head())

# data processing
df['gender'] = df['gender'].replace({'man': 1, 'woman': 2})
df.dropna(inplace = True)
df = df.drop(columns=['time'])

# check correlation between features
print(df.dtypes)
print(df.corr())


# personally I select height vs shoe_size given it has the strongest correlation
X = np.array(df['height']).reshape(-1, 1)
y = np.array(df['shoe_size']).reshape(-1, 1)


# splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# training the ML model
regr = LinearRegression()
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))


# predicted result
y_pred = regr.predict(X_test)
result_dict = {"height": X_test.tolist(), "actual_shoe_size": y_test.tolist() ,"predict_shoe_size": y_pred.tolist()}
result_df = pd.DataFrame.from_dict(result_dict)
# print(result_df)


# saving ml model
import pickle
filename = 'shoe_size_prediction_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regr, file)

