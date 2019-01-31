# Part 1 - Data Preprocessing

# Importing the libraries

import time 
start_time = time.time() 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray() # These 2 lines create dummy variables
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Import keras
import keras
from keras.models import Sequential
from keras.layers import Dense # The hidden layers
from keras.layers import Dropout

# Set hyperparameters
dropout_rate = 0.452
batch_size = 19
layers = 4
epochs = 126
number_of_neurons = 51
optimizer = 'adam'

#classifier = Sequential()  
#classifier.add(Dense(number_of_neurons, kernel_initializer='uniform', activation='relu', input_shape=(11,))) 
#classifier.add(Dropout(rate=dropout_rate))
#for _ in range(layers):
#	classifier.add(Dense(number_of_neurons, kernel_initializer='uniform', activation='relu'))
#	classifier.add(Dropout(rate=dropout_rate)) 
#classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']) 

# Fit the ANN to the training set
#classifier.fit(X_train, y_train, batch_size=64, epochs=500)
#
## Predicting the Test set results
#y_pred = classifier.predict(X_test)
#
## Convert to the binary outcomes so it can be used in the confusion matrix
#y_pred = (y_pred > 0.5)

# Evaluate
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score

# Wrap Keras functionality to use sklearn's K-Fold CV capabilities. 
def build_classifier(): # Needed for KerasClassifier
    classifier = Sequential()  
    classifier.add(Dense(number_of_neurons, kernel_initializer='uniform', activation='relu', input_shape=(11,))) 
    classifier.add(Dropout(rate=dropout_rate))
    for _ in range(layers):
        classifier.add(Dense(number_of_neurons, kernel_initializer='uniform', activation='relu'))
        classifier.add(Dropout(rate=dropout_rate)) 
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']) 
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=batch_size, epochs=epochs)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=5, n_jobs=-1)

mean = accuracies.mean()
variance = accuracies.std()

print("Mean: " + str(mean))
print("Variance: " + str(variance))

print("--- %s seconds ---" % (time.time() - start_time))

