import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import svm
# from sklearn import tree
from xgboost import XGBClassifier

dr = pd.read_csv("D:/DR ML Program/DR.csv")

# split data into train and test sets
xtrain, xtest, ytrain, ytest = train_test_split(dr.iloc[:,0:19], dr["result"], test_size=0.2, random_state=42)
# fit model no training data
model = XGBClassifier()
model.fit(xtrain, ytrain)
# make predictions for test data
ypred = model.predict(xtest)
predictions = [round(value) for value in ypred]
# evaluate predictions
accuracy = accuracy_score(ytest, predictions)
print("Accuracy: %.2f%%" % (accuracy*100.0))
