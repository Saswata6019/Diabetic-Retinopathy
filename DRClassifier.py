import numpy as np 
import pandas as pd 
import tensorflow as tf

dr = pd.read_csv("D:/DR ML Program/DR.csv")
predict = pd.read_csv("D:/DR ML Program/predict.csv")

from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(dr.iloc[:,0:19], dr["result"], test_size = 0.25, random_state = None)

columns = dr.columns[0:19]
 
feature_columns = [tf.contrib.layers.real_valued_column(k) for k in columns]

def input_fn(df,labels):
    feature_cols = {k:tf.constant(df[k].values, shape = [df[k].size,1]) for k in columns}
    label = tf.constant(labels.values, shape = [labels.size,1])
    return feature_cols,label

classifier = tf.contrib.learn.DNNClassifier(
	feature_columns = feature_columns,
	hidden_units = [15],
	n_classes = 2,
	optimizer = tf.train.AdamOptimizer(learning_rate=0.00146, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-4),
  	activation_fn = tf.nn.sigmoid
)

classifier.fit(input_fn = lambda:input_fn(xtrain,ytrain), steps = 1000)
ev = classifier.evaluate(input_fn = lambda:input_fn(xtest,ytest), steps = 1)
print(ev)

def input_predict(df):
    feature_cols = {k:tf.constant(df[k].values, shape = [df[k].size,1]) for k in columns}
    return feature_cols

pred = classifier.predict_classes(input_fn = lambda:input_predict(predict))
print(list(pred))
