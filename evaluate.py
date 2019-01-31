# Part 1 - Data Preprocessing

# Importing the libraries

import time 
start_time = time.time() 

from handle_data import X_train, y_train, X_test, y_test

# Import keras
import keras
from keras.models import Sequential
from keras.layers import Dense # The hidden layers
from keras.layers import Dropout

from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score

# Set hyperparameters
dropout_rate = 0.452
batch_size = 19
layers = 4
epochs = 126
number_of_neurons = 51
optimizer = 'adam'

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

