# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
td_train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
dataset = pd.concat([td_train, test], ignore_index=True)

# Embarked
# only in titanic_df, fill the two missing values with the most occurred value, which is "S".
dataset["Embarked"] = dataset["Embarked"].fillna("S")
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean()) # Impute missing age values with mean age
dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].mean())
dataset['Survived'].fillna(value=0,inplace=True)

# drop the unused variables
X = dataset.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1).values
y = dataset.iloc[:, 10].values

'''
# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(y[:, 0])
y[:, 0] = imputer.transform(y[:, 0])'''

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
onehotencoder = OneHotEncoder(categorical_features = [1]) # create dummy variables
X = onehotencoder.fit_transform(X).toarray()
onehotencoder = OneHotEncoder(categorical_features = [0]) # create dummy variables
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)