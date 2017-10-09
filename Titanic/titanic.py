# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
td_train = pd.read_csv('train.csv')
test = pd.read_csv('train.csv')
full_data = ['td_train', 'test']

# Embarked
# only in titanic_df, fill the two missing values with the most occurred value, which is "S".
td_train["Embarked"] = td_train["Embarked"].fillna("S")
td_train['Age'] = td_train['Age'].fillna(td_train['Age'].mean()) # Impute missing age values with mean age

#test data
test["Embarked"] = test["Embarked"].fillna("S")
test['Age'] = test['Age'].fillna(td_train['Age'].mean())

# drop the unused variables
X = td_train.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1).values
y = td_train.iloc[:, 1].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [1]) # create dummy variables
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
X_train = X
y_train = y
#test
X_test = test.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1).values

labelencoder_X = LabelEncoder()
X_test[:, 1] = labelencoder_X.fit_transform(X_test[:, 1])
X_test[:, 6] = labelencoder_X.fit_transform(X_test[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [1]) # create dummy variables
X_test = onehotencoder.fit_transform(X_test).toarray()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'poly', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#X_train = sc.inverse_transform(X_train) # transform back
#X_test = sc.inverse_transform(X_test) 

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#XX1 = X_set[:, 2]
#XX2 = X_set[:, 3]
#X1, X2 = np.meshgrid(XX1, XX2)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

'''Z_train = np.array([X1.ravel(), X2.ravel()]).T
plt.contourf(X1, X2, classifier.predict(sc.transform(Z_train)).reshape(X1.shape),  # TRANFORM Z
                                    alpha=0.75,
                                    cmap=ListedColormap(
                                    ('red', 'green')))'''
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logestic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Survived')
plt.legend()
plt.show()