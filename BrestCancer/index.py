import pandas as pd 
import numpy as np 
import matplotlib.pyplot as pyplot

#importing Dataset
dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header = None)
#print(dataset)
from sklearn.preprocessing import LabelEncoder
X = dataset.loc[:, 2:].values
y = dataset.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
#print(y)
#split the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

# Fitting classifier to the Training set
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# save the model to disk
import pickle
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))