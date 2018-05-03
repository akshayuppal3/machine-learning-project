import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
##feature selection
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
#models
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
#grid search
from sklearn.model_selection import GridSearchCV
#pipiline
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
#plot
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
#warnings
import warnings

#Suppress warning
def warn(*args, **kwargs):
    pass

#@return accuarcy of the model
def dt_best_features(X_train,Y_train,X_test, Y_test):
    clf = ExtraTreesClassifier()
    clf = clf.fit(X_train, Y_train)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X_train)
    X_test = model.transform(X_test)
    # baseline_models(X_new,Y_train,X_dev_new,Y_dev)
    Y_pred,score,_ = decision_tree(X_train,Y_train,X_test,Y_test)
    return(score)

#baseline models
def perceptron(X_train,Y_train,X_test,Y_test):
    clf = Perceptron(tol= None,random_state = 4)
    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    score = accuracy_score(Y_test,Y_pred)
    return(Y_pred,score,clf)

def decision_tree(X_train,Y_train,X_test,Y_test):
    param_grid = [{'max_depth':np.arange(2,50)}]
    tree = GridSearchCV(DecisionTreeClassifier(),param_grid) 
    tree.fit(X_train,Y_train)
    Y_pred = tree.predict(X_test)
    score = accuracy_score(Y_test,Y_pred)
    #print("Accuracy : decision_tree: ", score)
    #score = classification(Y_test,Y_pred)
    return(Y_pred,score,dt)

def baseline_models(X_train,Y_train,X_test,Y_test):
    print("Running the baseline models")    
    Y_pred_perc,perc_score = perceptron(X_train,Y_train,X_test,Y_test)
    print("Accuracy : perceptron: ", perc_score)
    classification(Y_test,Y_pred_perc)
    

def classification(Y_test,Y_pred):
    confusion_matrix(Y_test,Y_pred)
    print('  Classification Report:\n',classification_report(Y_test,Y_pred),'\n')

#better models
def svm_wrapper(X_train,Y_train,X_test,Y_test):
    param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001, 0.0001], 'kernel': ['rbf']},]
    clf = GridSearchCV(SVC(),param_grid)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    score = accuracy_score(Y_test,Y_pred)
    return(score,Y_pred,clf)

def voting_classifier(X_train,Y_train,X_test,Y_test):
    _,_,dt = decision_tree(X_train,Y_train,X_test,Y_test)
    _,_,perc = perceptron(X_train,Y_train,X_test,Y_test)
    _,_,svm = svm_wrapper(X_train,Y_train,X_test,Y_test)
    eclf = VotingClassifier(estimators=[('dt', dt),('svc', svm),('perc',perc)], voting='hard', weights=[1,3,3]) #('svc', svm),('knn', knn)
    eclf.fit(X_train, Y_train)
    Y_pred = eclf.predict(X_test)
    score1 = accuracy_score(Y_test,Y_pred)
    return(score1)

#Changed the logic with grid search
def mlp_wrapper(X_train,Y_train, X_test,Y_test):
    param_grid = [{'hidden_layer_sizes':np.arange(10,150,10),'max_iter':[30] ,'activation':['logistic', 'tanh','relu']}]
    mlp = GridSearchCV(MLPClassifier(), param_grid)
    mlp.fit(X_train, Y_train)
    Y_pred = mlp.predict(X_test)
    score = accuracy_score(Y_test,Y_pred)
    return(score,Y_pred)

#from sklearn.preprocessing import CategoricalEncoder
def knn(X_train,Y_train, X_test,Y_test):
    param_grid = [{'n_neighbors': np.arange(2,250,20)}]
    knn = GridSearchCV(KNeighborsClassifier(),param_grid) 
    knn.fit(X_train,Y_train)
    Y_pred = knn.predict(X_test)
    score = (np.mean(Y_pred == Y_test))
    return(score,Y_pred)

def naive_bayes(X_train,Y_train, X_test, Y_test):
    clf = BernoulliNB()
    clf.fit(X_train, Y_train)
    Y_pred = neigh.predict(X_test)
    score = accuracy_score(Y_test,Y_pred)
    return(score,Y_pred)

def extended_model(X_train,Y_train,X_test,Y_test):
    score,Ypred =knn(X_train,Y_train, X_test,Y_test)
    print("Accuracy : knn ", score)
    score, Y_pred = mlp_wrapper(X_train, Y_train, X_test, Y_test, 1500, 'relu')
    print("Accuracy : MLP ", score)
    score, Y_pred = svm_wrapper(X_train,Y_train,X_test,Y_test)
    print("Accuracy: SVM ", score)
    score, Y_pred = bagging(X_train,Y_train, X_test, Y_test)
    print("Accuracy: bagging:  ", score)
#v1
def bagging_with_tree(X_train,Y_train,X_test,Y_test,depth):
    _,_,dt =decision_tree(X_train,Y_train,X_test,Y_test,depth)
    BaggingClassifier(n_estimators = n_estimator,random_state= 4,base_estimator = dt)
    bag.fit(X_train, Y_train)
    Y_pred = bag.predict(X_test)
    score = accuracy_score(Y_test,Y_pred)
    return(score)       
#v2: Current version
def bagging_with_DT(X_train,Y_train, X_test, Y_test):
    for i in range(1,5):
        dt = DecisionTreeClassifier(max_depth = i,random_state = 4)
        param_grid = [{'n_estimators': np.arange(1,30),'base_estimator':[dt]}]
        bag = GridSearchCV(BaggingClassifier(), param_grid)
        bag.fit(X_train, Y_train)
        Y_pred = bag.predict(X_dev)
    score.append(accuracy_score(Y_dev,Y_pred))
    sc = np.sort(score)[::-1][0]
    print("bagging with decision tree:", sc)
    # return(score,Y_pred)        

def boosting(X_train,Y_train, X_test, Y_test):
    gb= GradientBoostingClassifier()
    gb.fit(X_train, Y_train)
    Y_pred = bag.predict(X_test)
    score = np.mean(gb.predict(X_dev) == Y_dev)
    return(score,Y_pred)

def PCA_with_DT(X_train,Y_train, X_test, Y_test):
    score = [0]
    for i in range(1,20):
        estimators1 = [('tree', PCA()), ('clf', DecisionTreeClassifier(max_depth =i,random_state = 4) )]
        pipe1 = Pipeline(estimators1)
        pipe1.fit(X_train,Y_train)
        Y_pred = pipe1.predict(X_dev)
        score.append(np.mean(Y_pred == Y_dev))
    sc = np.sort(score)[::-1][0]
    print("PCA with DT:", sc )

def PCA_with_SVM(X_train,Y_train, X_test, Y_test):
    estimators = [('reduce_dim', PCA()), ('clf', SVC() )]
    pipe = Pipeline(estimators)
    pipe.fit(X_train,Y_train)
    Y_pred = pipe.predict(X_dev)
    score = (np.mean(Y_pred == Y_dev))
    print(score)
