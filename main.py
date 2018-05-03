from preprocess import *
import data
from util import *
import argparse
import sys
import preprocess as pr


# def preprocess():
# 	x = pr.Preprocess()
# 	#preprocessing
# 	X_train,Y_train,X_val,Y_val,X_test,Y_test = x.get_data()


 #(Perceptron, knn, dt) measures baseline acc and classification report @return void
def train_baseline(X_train,Y_train,X_dev,Y_dev,X_test,Y_test):	
	models = baseline_models(X_train,Y_train)
	print("Development accuracy")
	prediction_models(models,X_dev,Y_dev)
	print("\n" * 3)
	print("Testing accuracy")
	prediction_models(models,X_test,Y_test)

#This takes some time
def train_extended(X_train,Y_train,X_dev,Y_dev,X_test,Y_test):
	models = extended_model(X_train,Y_train)
	prediction_models(models,X_dev,Y_dev)
	print("\n" * 3)
	print("Testing accuracy")
	prediction_models(models,X_test,Y_test)

def train_ensemble(X_train,Y_train,X_dev,Y_dev,X_test,Y_test):
	print("Trying ensemble_models")
	ensemble_models(X_train,Y_train,X_dev,Y_dev,X_test,Y_test)

def main():
	#filter warnings
	warnings.warn = warn
	x = pr.Preprocess()
	#preprocessing
	X_train,Y_train,X_dev,Y_dev,X_test,Y_test = x.get_data()

	# parser = argparse.ArgumentParser(description ='IML prog 5')
	# #group = parser.add_mutually_exclusive_group(required=True)
	# #parser.add_argument('-p', '--prep', help='to preprocess the data', default = 'prep',required = False)
	# parser.add_argument('-t', '--train', help='to run the baseline_models', default = 'train', required = False)
	# args = vars(parser.parse_args())
	# if args['prep'] == 'prep':
	# 	preprocess()	
	# if args['train'] == 'all':
	# 	all()

	#Using sys agr
	if(sys.argv[1] == 'train_b'):
		train_baseline(X_train,Y_train,X_dev,Y_dev,X_test,Y_test)
	elif(sys.argv[1] == 'train_e'):
		print(X_train.shape,X_dev.shape)
		train_extended(X_train,Y_train,X_dev,Y_dev,X_test,Y_test)

if (__name__ == '__main__'):
	main()