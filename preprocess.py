from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Normalizer
import numpy as np
from data import *
import data

class Preprocess:
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
        #Coverting the Y to array
        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()
        Y_val = Y_val.ravel()
        #Normalizing the features
        norm = Normalizer()
        norm.fit(X_train,Y_train)
        X_train = norm.transform(X_train)
        X_val = norm.transform(X_val)
        X_test = norm.transform(X_test)
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