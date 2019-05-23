import autosklearn.classification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings 
import sklearn.metrics 

scores = pd.read_csv("API_Scores.csv")
data,labels = np.split(scores, [4], axis=1)
print(data)
print(labels)

x_train, x_test, y_train, y_test = train_test_split(data,labels, test_size=0.20)
print("AutoSklearn Implementation")
cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=3600)

with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	cls.fit(x_train, y_train)

predict_train = cls.predict(x_train)
predict_test = cls.predict(x_test)

print("accuracy on training set: %f" % sklearn.metrics.accuracy_score(y_train, predict_train))
print("accuracy on test set: %f" % sklearn.metrics.accuracy_score(y_test, predict_test))

print(cls.sprint_statistics())

# Best performing model with its hyperparameters

cls.cv_results_['params'][np.argmax(cls.cv_results_['mean_test_score'])]


print(cls.show_models())	
