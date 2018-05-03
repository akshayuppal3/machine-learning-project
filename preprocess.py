from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
import numpy as np
from data import *
import data

class Preprocess:
# Function to take care of missing values
    # Function not used: Changed to preprocessing
    def missing_values(X, y):
        imp = Imputer(strategy= 'most_frequent', axis = 0) 
        imp.fit(X)
        X = imp.transform(X)
        imp.fit(y.reshape(-1,1))
        y = imp.transform(y.reshape(-1,1))   
        return(X,y)

    #Fucntion not active
    def OneHot(X_train,Y_train, X_test, Y_test, X_val, Y_val):
        enc = OneHotEncoder()   #sparse = 'False',n_values= (np.max(X,axis= 0) + 1))
        enc.fit(X_train)
        X_train = enc.transform(X_train).toarray()
        X_test = enc.transform(X_test).toarray()
        X_val = enc.transform(X_val).toarray()
        #Coverting the Y to array
        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()
        Y_val = Y_val.ravel()
        return(X_train,Y_train, X_test, Y_test, X_val, Y_val)

    #removes null values and performs one hot encoding
    def preprocessing(self,X_train,Y_train, X_test, Y_test, X_val, Y_val):
        print("preprocessing data")
        imp = Imputer(strategy= 'most_frequent', axis = 0) 
        imp.fit(X_train)
        X_train = imp.transform(X_train)
        X_val = imp.transform(X_val)
        X_test = imp.transform(X_test)
        imp.fit(Y_train.reshape(-1,1))
        Y_train = imp.transform(Y_train.reshape(-1,1))
        Y_test = imp.transform(Y_test.reshape(-1,1))
        Y_val = imp.transform(Y_val.reshape(-1,1)) 
        #Changing categoriral features to nominal
        return(X_train,Y_train, X_test, Y_test, X_val, Y_val)
    
    def get_data(self):
        return(self.X_train,self.Y_train,self.X_val,self.Y_val,self.X_test,self.Y_test)
 
    def __init__(self):
        x = data.Load()
        #preprocessing
        X_train,Y_train,X_val,Y_val,X_test,Y_test = x.get_data()
        X_train,Y_train, X_test, Y_test, X_val, Y_val = self.preprocessing(X_train,Y_train, X_test, Y_test, X_val, Y_val)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test