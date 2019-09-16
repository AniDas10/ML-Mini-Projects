import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("insurance_data.csv")
X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, test_size=0.2)
model = LogisticRegression()

model.fit(X_train, y_train)
model.predict(X_test)
model.score(X_test, y_test)


