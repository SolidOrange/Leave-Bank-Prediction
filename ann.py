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

# Set hyperparameters
dropout_rate = 0.452
batch_size = 19
layers = 4
epochs = 126
number_of_neurons = 51
optimizer = 'adam'

classifier = Sequential()  
classifier.add(Dense(number_of_neurons, kernel_initializer='uniform', activation='relu', input_shape=(11,))) 
classifier.add(Dropout(rate=dropout_rate))
for _ in range(layers):
	classifier.add(Dense(number_of_neurons, kernel_initializer='uniform', activation='relu'))
	classifier.add(Dropout(rate=dropout_rate)) 
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']) 

#Fit the ANN to the training set
classifier.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Convert to the binary outcomes so it can be used in the confusion matrix
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix: ")
print(str(cm))

print("--- %s seconds ---" % (time.time() - start_time))


