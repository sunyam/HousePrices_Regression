import pandas as pd
import numpy as np

# Read dataset into X and Y
df = pd.read_csv('/Users/sunyambagga/Github/HousePrices_Regression/housing.csv', delim_whitespace=True, header=None)
dataset = df.values

X = dataset[:, 0:13]
Y = dataset[:, 13]

#print "X: ", X
#print "Y: ", Y


# Define the neural network
from keras.models import Sequential
from keras.layers import Dense

def build_nn():
    model = Sequential()
    model.add(Dense(20, input_dim=13, init='normal', activation='relu'))
    # No activation needed in output layer (because regression)
    model.add(Dense(1, init='normal'))

    # Compile Model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# Evaluate model (kFold cross validation)
from keras.wrappers.scikit_learn import KerasRegressor

# sklearn imports:
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Before feeding the i/p into neural-network, standardise the dataset because all input variables vary in their scales
estimators = []
estimators.append(('standardise', StandardScaler()))
estimators.append(('multiLayerPerceptron', KerasRegressor(build_fn=build_nn, nb_epoch=100, batch_size=5, verbose=0)))

pipeline = Pipeline(estimators)

kfold = KFold(n=len(X), n_folds=10)
results = cross_val_score(pipeline, X, Y, cv=kfold)

print "Mean: ", results.mean()
print "StdDev: ", results.std()