# machine-learning-project
--------------------------------------------SETUP--------------------------------------------------------
1) Extract the zip file and store it locally.(root/)
2) Download the dataset "columns.csv" and "responses.csv" from "https://www.kaggle.com/miroslavsabo/young-people-survey/data" and save the file in the same directory of project files (root/machine-learning-project).
3) Open a new terminal window/tab
4) Navigate to the local repository where file is saved.(root/machine-learning-project) 
5) The code consists of 3 parts and it can be run independently of each other:
    5 a) Baseline models: It will preprocess the data and will run the baseline models (decision tree, knn, perceptron).It will display the accuracy and precision, recall of each model.
	Run with following command: python main.py train_b
	estimated time : 42s
	5 b) Extended models: It will preprocess and run on the models, (naive_bayes,MLP,PCA_with_SVM and random_forest) that I tried to get better accuracy than the baseline models. It will display the accuracy and precision , recall of each model.
	Run command: python main.py train_e
	estimated time: 2min 10s
	5 c) Hypertuning of random forest: This includes the tuning the hyperparameters of random forest and performing L2 regularization to select the best features. 
	Run command: python main.py hyper_tune
	estimated time: 2min 1s
	
