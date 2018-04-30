import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
##feature selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# Function to take care of missing values
def missing_values(X, y):
    imp = Imputer(strategy= 'most_frequent', axis = 0) 
    imp.fit(X)
    X = imp.transform(X)
    imp.fit(y.reshape(-1,1))
    y = imp.transform(y.reshape(-1,1))   
    return(X,y)

# Function to take care of missing values
def missing_values(X, y):
    imp = Imputer(strategy= 'most_frequent', axis = 0) 
    imp.fit(X)
    X = imp.transform(X)
    imp.fit(y.reshape(-1,1))
    y = imp.transform(y.reshape(-1,1))   
    return(X,y)

#One hot encoding function
def preprocessing(X,Y):
    imp = Imputer(strategy= 'most_frequent', axis = 0) 
    imp.fit(X)
    X = imp.transform(X)
    imp.fit(Y.reshape(-1,1))
    Y = imp.transform(Y.reshape(-1,1))
    #Changing categoriral features to nominal
    enc = OneHotEncoder(sparse = 'False',n_values= (np.max(X,axis= 0) + 1))
    enc.fit(X)
    X = enc.transform(X).toarray()
    return(X,Y)

#baseline models
def perceptron(X_train,Y_train,X_test,Y_test):
	clf = Perceptron(random_state = 4)
	clf.fit(X_train,Y_train)
	Y_pred = clf.predict(X_test)
	accuracy_score(Y_test,Y_pred)


