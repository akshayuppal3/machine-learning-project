from data import *
import data
from util import *
import argparse

def models():
	#Checking fo rthe baseline models
	# print("function called")
	baseline_models(X_train,Y_train,X_val,Y_val)
	extended_model(X_train,Y_train,X_test,Y_test)

def preprocess():
	#loading the data
	x = data.Load()
	#preprocessing
	X_train,Y_train,X_val,Y_val,X_test,Y_test = x.get_data()
	X_train,Y_train, X_test, Y_test, X_val, Y_val = preprocessing(X_train,Y_train, X_test, Y_test, X_val, Y_val)

def all():
	x = data.Load()
	#preprocessing
	X_train,Y_train,X_val,Y_val,X_test,Y_test = x.get_data()
	X_train,Y_train, X_test, Y_test, X_val, Y_val = preprocessing(X_train,Y_train, X_test, Y_test, X_val, Y_val)
	X_train,Y_train, X_test, Y_test, X_val, Y_val = OneHot(X_train,Y_train, X_test, Y_test, X_val, Y_val)
	baseline_models(X_train,Y_train,X_val,Y_val)
	extended_model(X_train,Y_train,X_test,Y_test)
	bagging_with_tree(X_train,Y_train,X_test,Y_test,depth)

def main():
	#filter warnings
	warnings.warn = warn

	parser = argparse.ArgumentParser(description ='IML prog 5')
	#group = parser.add_mutually_exclusive_group(required=True)
	#parser.add_argument('-p', '--prep', help='to preprocess the data', default = 'prep',required = False)
	parser.add_argument('-t', '--train', help='to run the baseline_models', default = 'train', required = False)
	args = vars(parser.parse_args())
	# if args['prep'] == 'prep':
	# 	preprocess()	
	if args['train'] == 'all':
		all()

if (__name__ == '__main__'):
	main()