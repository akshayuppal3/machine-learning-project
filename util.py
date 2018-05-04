import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
##feature selection
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
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

#testing function
#to do create a function that would do the prediction
def prediction_models(models,X,Y):
    for name,model in models.items():
        Y_pred = model.predict(X)
        score = accuracy_score(Y,Y_pred)
        print("accuarcy ",name,":", score)
        classification(Y,Y_pred)

def prediction(model,X,y):
    Y_pred = model.predict(X)
    score = accuracy_score(y,Y_pred)
    return(score)

def classification(Y_dev,Y_pred):
    confusion_matrix(Y_dev,Y_pred)
    print('  Classification Report:\n',classification_report(Y_dev,Y_pred),'\n')


#baseline models
def perceptron(X_train,Y_train):
    clf = Perceptron(tol= None,random_state = 4)
    clf.fit(X_train,Y_train)
    return(clf)

def decision_tree(X_train,Y_train):
    param_grid = [{'max_depth':np.arange(2,50)}]
    tree = GridSearchCV(DecisionTreeClassifier(),param_grid) 
    tree.fit(X_train,Y_train)
    return(tree)

#from sklearn.preprocessing import CategoricalEncoder
def knn(X_train,Y_train):
    param_grid = [{'n_neighbors': np.arange(2,250,20)}]
    knn = GridSearchCV(KNeighborsClassifier(),param_grid) 
    knn.fit(X_train,Y_train)    
    return(knn)

#better models
def svm_wrapper(X_train,Y_train):
    param_grid = [
    {'C': [1, 10], 'kernel': ['linear']},
    {'C': [1, 10], 'gamma': [0.1,0.01], 'kernel': ['rbf']},]
    svm1 = GridSearchCV(SVC(),param_grid)
    svm1.fit(X_train, Y_train)
    return(svm1)

def voting_classifier(X_train,Y_train):
    dt = decision_tree(X_train,Y_train)
    perc = perceptron(X_train,Y_train)
    svm = svm_wrapper(X_train,Y_train)
    eclf = VotingClassifier(estimators=[('dt', dt),('svc', svm),('perc',perc)], voting='hard', weights=[1,3,3]) #('svc', svm),('knn', knn)
    eclf.fit(X_train, Y_train)
    return(eclf)

#Changed the logic with grid search
def mlp_wrapper(X_train,Y_train):
    param_grid = [{'hidden_layer_sizes':np.arange(10,250,25),'max_iter':[30] ,'activation':['logistic', 'tanh','relu']}]
    mlp = GridSearchCV(MLPClassifier(), param_grid)
    mlp.fit(X_train, Y_train)
    return(mlp)
#b--
def naive_bayes(X_train,Y_train):
    clf = BernoulliNB()
    clf.fit(X_train, Y_train)
    return(clf)

def baseline_models(X_train,Y_train):
    models = {}
    perc = perceptron(X_train,Y_train)
    knearest = knn(X_train,Y_train)
    dt = decision_tree(X_train,Y_train)
    models = {"perceptron":perc,"knn":knearest,"decision_tree":dt}
    return(models)

#@return the models @params training data
def extended_model(X_train,Y_train,X_dev, Y_dev):
    models= {}
    nb= naive_bayes(X_train,Y_train)
    mlp = mlp_wrapper(X_train, Y_train)
    svm2 = PCA_with_SVM(X_train,Y_train)
    dt = PCA_with_DT(X_train,Y_train,X_dev, Y_dev)
    rd = random_forest(X_train,Y_train,X_dev,Y_dev)
    models = {"naive_bayes":nb,"MLP":mlp,"PCA_with_SVM":svm2,"random_forest":rd}
    return(models)

def ensemble_models(X_train,Y_train,X_dev,Y_dev,X_test,Y_test):
    bagging_with_DT(X_train,Y_train, X_dev, Y_dev)

def boosting(X_train,Y_train):
    gb= GradientBoostingClassifier()
    gb.fit(X_train, Y_train)
    return(gb)

#
def PCA_with_DT(X_train,Y_train,X_dev, Y_dev):
    score = [0]
    for i in range(1,20):
        estimators1 = [('tree', PCA()), ('clf', DecisionTreeClassifier(max_depth =i,random_state = 4) )]
        pipe1 = Pipeline(estimators1)
        pipe1.fit(X_train,Y_train)
        Y_pred = pipe1.predict(X_dev)
        score.append(np.mean(Y_pred == Y_dev))
    sc = np.sort(score)[::-1][0]
    print("PCA with DT:", sc )

##@returns PCA_with_svm model
def PCA_with_SVM(X_train,Y_train):
    estimators = [('reduce_dim', PCA()), ('clf', SVC() )]
    pipe = Pipeline(estimators)
    pipe.fit(X_train,Y_train)
    return(pipe)

def random_forest(X_train,Y_train,X_dev,Y_dev):
    score1 = []
    model = []
    for i in np.arange(1,10):
        for j in np.arange(1,30):
            rand = RandomForestClassifier(max_depth = i, n_estimators = j,random_state = 4,n_jobs = -1)
            rand.fit(X_train,Y_train)
            Y_pred = rand.predict(X_dev)
            score = accuracy_score(Y_pred,Y_dev)
            score1.append(score)
            model.append(rand)
    sc = np.sort(score1)[::-1][0]
    idx= np.argsort(score1)[::-1][0]
    rand_model = model[idx]
    return(rand_model)

def feature_engineering(X_train,Y_train,X_dev,Y_dev,i):
    KBest = SelectKBest(chi2,k=i)
    KBest.fit(X_train,Y_train)
    X_train = KBest.transform(X_train)
    X_dev = KBest.transform(X_dev)
    return(X_train,Y_train,X_dev,Y_dev,KBest)

def regularization(X_train,Y_train,X_dev,Y_dev,c):
    lv = LinearSVC(penalty="l2",C=c,dual=False)
    lv.fit(X_train,Y_train)
    model = SelectFromModel(lv, prefit=True)
    X_tr_new = model.transform(X_train)
    X_dv_new = model.transform(X_dev)
    rand_model =random_forest(X_tr_new,Y_train,X_dv_new,Y_dev)
    score = prediction(rand_model,X_dv_new,Y_dev)
    return(score,rand_model,model)

#return acccuracy of testing
def feature_tuning_rf(X_train,Y_train,X_dev,Y_dev,X_test,Y_test):
    params_C = [0.01,0.1,1]
    scores = []
    rf_model = []
    tf_model = []
    for c in params_C:
        score,rand,model = regularization(X_train,Y_train,X_dev,Y_dev,c)
        scores.append(score)
        rf_model.append(rand)
        tf_model.append(model)
    idx = np.argsort(score)[::-1][0]
    rand_model_new = rf_model[idx]
    tf_model_new = tf_model[idx]
    #development accuracy
    X_dev_new = tf_model_new.transform(X_dev)
    score = prediction(rand_model_new,X_dev_new, Y_dev)
    print("development acccuracy with L2 regularization: ", score)
    Y_dev_pred = rand_model_new.predict(X_dev_new)
    classification(Y_dev,Y_dev_pred)  
    X_test_new = tf_model_new.transform(X_test)
    score = prediction(rand_model_new,X_test_new,Y_test)
    print("testing acccuracy with L2 regularization: ", score)
    Y_pred = rand_model_new.predict(X_test_new)
    classification(Y_test,Y_pred)
    
#@return testing data and model of the model
def dt_best_features(X_train,Y_train,X_dev,Y_dev,X_test,Y_test):
    clf = ExtraTreesClassifier()
    clf = clf.fit(X_train, Y_train)
    model = SelectFromModel(clf, prefit=True)
    X_tr_new = model.transform(X_train)
    X_dv_new = model.transform(X_dev)
    tree = decision_tree(X_tr_new,Y_train)
    return(X_test,Y_test,tree)

#v2: Current version
def bagging_with_DT(X_train,Y_train, X_dev, Y_dev):
    score =[]
    model = []
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