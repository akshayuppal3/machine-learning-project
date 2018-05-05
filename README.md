# machine-learning-project
HW5 Extra credit for CS412 – University of Illinois at Chicago Name: auppal8@uic.edu
--------------------------------------------SETUP--------------------------------------------------------
1) Extract the zip file and store it locally.(root/)
2) Download the dataset "columns.csv" and "responses.csv" from "https://www.kaggle.com/miroslavsabo/young-people-survey/data" and save the file in the same directory of project files (root/machine_learning-project). If you choose download all dataset from kaggle then please extract both of the files at the same directory.(root/machine_learning-project)
3) Open a new terminal window/tab
4) Navigate to the local repository where file is saved.(root/machine_learning-project) 
5) The code consists of 3 parts and it can be run independently of each other:
	a) Baseline models: It will preprocess the data and will run the baseline models (decision tree, knn, perceptron).It will display the accuracy and precision, recall of each model.
	Run with following command: python main.py train_b

	b) Extended models: It will preprocess and run on the models, (naive_bayes,MLP,PCA_with_SVM and random_forest) that I tried to get better accuracy than the baseline models. It will display the accuracy and precision , recall of each model.
	Run command: python main.py train_e

	c) Hypertuning of random forest: This includes the tuning the hyperparameters of random forest and performing L2 regularization to select the best features. 
	Run command: python main.py hyper_tune
	estimated time: 2min 1s
	
