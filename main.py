import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib as plt

data  = pd.read_csv("heart.csv")

le = preprocessing.LabelEncoder()
cols = ["Sex","ExerciseAngina","ChestPainType","RestingECG","ST_Slope"]

for col in cols:
    data[col]= le.fit_transform(data[col])
    print(le.classes_)

print(data)

y = data["HeartDisease"]
X = data.drop(["HeartDisease"],axis=1)
X = (X - X.mean()) / X.std()

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=12,random_state=0)

from LogisticRegression import LogisticRegression

LR = LogisticRegression(lr=0.0001, n_iters=1000)

LR.fit(X_train, y_train)

predictions = LR.predict(X_test)

def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)

mse_value = mse(y_test, predictions)
print(mse_value)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))