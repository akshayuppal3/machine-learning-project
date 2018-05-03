from sklearn.model_selection import train_test_split
import pandas as pd
import util

class Load:
	def load_data(self):
		columns = pd.read_csv('columns.csv')
		responses = pd.read_csv('responses.csv')
		item_list = list(columns['short'])	#get short column list
		index = item_list.index('Empathy')
		item_list_X = item_list[:index] + item_list[index+1 :]
		item_list_Y = item_list[index]
		X_all = responses.filter(items = item_list_X)
		Y_all = responses['Empathy']
		#Changing the object types to numeric
		my_list = list(X_all.select_dtypes(include=['object']).columns)
		X_all = pd.get_dummies(X_all,columns= my_list)
		# #dropping the fields with text in it
		# X_all =  X_all[X_all.dtypes[(X_all.dtypes=="float64")|(X_all.dtypes=="int64")].index.values]
		X_all = X_all.values  #converting to array
		Y_all = Y_all.values
		return(X_all, Y_all)

	def split_data(self,X,Y):
		X_train,X_test, Y_train, Y_test =  train_test_split(X, Y,test_size =0.2,random_state= 4 )
		X_train,X_val , Y_train, Y_val  = train_test_split(X_train, Y_train,test_size =0.25,random_state= 4 )
		return (X_train, Y_train , X_val, Y_val, X_test, Y_test)

	def __init__(self):
		X_all, Y_all = self.load_data()
		X_train, Y_train, X_val, Y_val, X_test, Y_test = self.split_data(X_all, Y_all)
		self.X_train = X_train
		self.Y_train = Y_train
		self.X_val = X_val
		self.Y_val = Y_val
		self.X_test = X_test
		self.Y_test = Y_test

	def get_data(self):
		return(self.X_train,self.Y_train,self.X_val,self.Y_val,self.X_test,self.Y_test)
		#return(X_train,Y_train,X_val,Y_val,X_test,Y_test)

#use class to hold data
