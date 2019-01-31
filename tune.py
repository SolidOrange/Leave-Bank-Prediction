
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

from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform

def build_classifier(optimizer, number_of_neurons, dropout_rate, layers): # Needed for KerasClassifier
    classifier = Sequential()  
    classifier.add(Dense(number_of_neurons, kernel_initializer='uniform', activation='relu', input_shape=(11,))) 
    classifier.add(Dropout(rate=dropout_rate))
    for _ in range(layers):
        classifier.add(Dense(number_of_neurons, kernel_initializer='uniform', activation='relu'))
        classifier.add(Dropout(rate=dropout_rate)) 
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']) 
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)

parameters = {
                'batch_size': sp_randint(1,100),
                'epochs':sp_randint(10,500),
                'optimizer':['adam','rmsprop'],
                'number_of_neurons': sp_randint(3,100),
                'dropout_rate': uniform(0.0,0.75),
                'layers': sp_randint(1,10)
              }

# Use Random Search instead of Grid Search
rs = RandomizedSearchCV(estimator=classifier, 
                        param_distributions=parameters,
                        n_iter=60,
                        scoring='accuracy',
                        cv=5,
                        verbose=2,
			n_jobs=-1
                        )
rs.fit(X_train, y_train)

# Evaluate the grid search
best_parameters = rs.best_params_
best_accuracy = rs.best_score_

print("Best Parameters: " + str(best_parameters))
print("Best Accuracy: " + str(best_accuracy))

print("--- %s seconds ---" % (time.time() - start_time))

