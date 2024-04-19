# This lab demonstrates use of simple KNN classification models on the telecustomer dataset. 

# import standard libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import warnings

# import sklearn libraries
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline # just included for future use, can comment out. 
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.feature_selection import SequentialFeatureSelector # for forward and backward selection

import optuna # for hyperparameter optimization
import joblib # for model save

warnings.filterwarnings('ignore')

## GET DATASET AND PREVIEW

file_path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv'
df = pd.read_csv(file_path)

# we want to show all columns when viewing the data
pd.set_option('display.max_columns', None)
df.head()
df.describe()
df.info
df.dtypes
df.shape

# Let’s see how many of each class is in our data set
df['custcat'].value_counts()
# 281 Plus Service, 266 Basic-service, 236 Total Service, and 217 E-Service customers

# explore other columns for visualization
df.hist(column='income', bins=50)
plt.close()

## FEATURE SET 
df.columns

# to use scikit-learn library, we have to convert pandas df to Numpy. 
#.values convert df to Numpy array

X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values  #.astype(float)
X[0:5]

# labels
y = df['custcat'].values
y[0:5]

## DATA NORMALIZATION
# Data Standardization gives the data zero mean and unit variance, it is good practice, 
# especially for algorithms such as KNN which is based on the distance of data points. 

X = StandardScaler().fit(X).transform(X.astype(float))
# StandardScaler().fit_transform(X.astype(float)) might be preferred 

X[0:5]

## Train Test Split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set shape:', X_train.shape,  y_train.shape)
print ('Test set shape:', X_test.shape,  y_test.shape)

## K-NEAREST NEIGHBOR CLASSIFIER

# 1 --- starting with k = 4. ---
k = 4

# Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

# Predicting on the test set
yhat = neigh.predict(X_test)
yhat[0:5]

print("Train set Accuracy using KNN:",accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy using KNN:", accuracy_score(y_test, yhat))

# check for probability of each class instead
# Predicting on the test set
yhat_proba = neigh.predict_proba(X_test)
yhat_proba[0:5]
class_labels = neigh.classes_

# access classes

# 2 --- trying for other k's ---

# What about other K?
# K in KNN, is the number of nearest neighbors to examine. It is supposed to be specified by the user.
# The general solution is to reserve a part of your data for testing the accuracy of the model, 
# then choose k =1, use the training part for modeling, and calculate the accuracy of prediction using all samples in your test set. 
# Repeat this process, increasing the k, and see which k is the best for your model.

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = accuracy_score(y_test, yhat)   
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc
# mean_acc

# Plot the model accuracy for a different number of neighbors.
plt.plot(range(1,Ks),mean_acc,'g') # g for color of line
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.savefig('model accuracy plot at different k.png')
plt.close()
# plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 

## OPTIMIZING HYPERPAREMETERS USING OPTUNA 

# recall: we have x_train, x_test, y_train, and y_test from above but let's reinstantiate them anyway

# Define objective function
def objective(trial):
    # Define search space for k (number of neighbors)
    k = trial.suggest_int('k', 1, 20)  # Searching integer values between 1 and 20
       
    # Define search space for p (Minkowski distance power)
    p = trial.suggest_float('p', 1.0, 2.0)  # Searching float values between 1.0 and 2.0

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=k, p=p)
    knn.fit(X_train, y_train)

    # Evaluate model
    y_pred = knn.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    return accuracy

# Set up Optuna study
study = optuna.create_study(direction='maximize')  # maximize accuracy
study.optimize(objective, n_trials=100)  # number of trials for optimization

# Get the best hyperparameters
best_params = study.best_params
best_accuracy = study.best_value

print("Best Hyperparameters k and p using Optuna:", best_params)
print("Best Accuracy using Optuna:", best_accuracy)

# use best params for prediction
neigh = KNeighborsClassifier(n_neighbors = best_params['k'], 
                             p = best_params['p']).fit(X_train,y_train)
neigh

# Predicting on the test set
yhat = neigh.predict(X_test)
yhat[0:5]

print("Train set Accuracy: ", accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", accuracy_score(y_test, yhat))

# check for probability of each class instead
# Predicting on the test set
yhat_proba = neigh.predict_proba(X_test)
yhat_proba[0:5]
class_labels = neigh.classes_

## SAVE MODEL
joblib.dump(neigh, 'knn-model.pkl')





