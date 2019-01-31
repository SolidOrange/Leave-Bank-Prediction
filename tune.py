
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

