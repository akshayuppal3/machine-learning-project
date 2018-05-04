from preprocess import *
import data
from util import *
import argparse
import sys
import preprocess as pr


 #(Perceptron, knn, dt) measures baseline acc and classification report @return void
def train_baseline(X_train,Y_train,X_dev,Y_dev,X_test,Y_test):	
	print("Training the baseline models(decision tree, knn, perceptron)")
	print("\n" *1)
	print("This might take some time :[estimated(42s)]")
	models = baseline_models(X_train,Y_train)
	print("Development accuracy")
	prediction_models(models,X_dev,Y_dev)
	print("\n" * 3)
	print("Testing accuracy")
	prediction_models(models,X_test,Y_test)

#This takes some time
def train_extended(X_train,Y_train,X_dev,Y_dev,X_test,Y_test):
	print("Trainig other models (naive_bayes,MLP,PCA_with_SVM and random_forest)")
	print("\n" *1)
	print("This might take some time :[estimated(2min 10s)]")
	models = extended_model(X_train,Y_train,X_dev, Y_dev)
	print("\n" * 2)
	print("Development accuracy")
	prediction_models(models,X_dev,Y_dev)
	print("\n" * 2)
	print("Testing accuracy")
	prediction_models(models,X_test,Y_test)
	print("\n" *1)
	print("""Since we are getting better development accuracy with Random forest so we will 
		tune the hyperarameters of random forest""")
	print("To tune pass the argument as 'hyper_tune'")

def feature_tuning(X_train,Y_train,X_dev,Y_dev,X_test,Y_test):
	print("Tuning the hyperparameters of random forest and performing L2 regularization to select the best features")
	print("\n" * 1)
	print("This might take some time :[estimated(2min 1s)]")
	feature_tuning_rf(X_train,Y_train,X_dev,Y_dev,X_test,Y_test)

def main():
	#filter warnings
	warnings.warn = warn
	x = pr.Preprocess()
	#preprocessing
	X_train,Y_train,X_dev,Y_dev,X_test,Y_test = x.get_data()

	#Using sys agr
	if(sys.argv[1] == 'train_b'):
		train_baseline(X_train,Y_train,X_dev,Y_dev,X_test,Y_test)
	elif(sys.argv[1] == 'train_e'):
		train_extended(X_train,Y_train,X_dev,Y_dev,X_test,Y_test)
	elif(sys.argv[1] == 'hyper_tune'):
		feature_tuning(X_train,Y_train,X_dev,Y_dev,X_test,Y_test)

if (__name__ == '__main__'):
	main()