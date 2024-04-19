# This lab demonstrates use of SVM using the cells dataset. 

# import standard libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import warnings

# import sklearn libraries
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, \
        confusion_matrix, classification_report, ConfusionMatrixDisplay, log_loss # for easier coding 
from sklearn.preprocessing import StandardScaler # for easier coding 
from sklearn.model_selection import train_test_split
from sklearn import svm # for support vector machine
import optuna

warnings.filterwarnings('ignore')

## GET DATASET AND PREVIEW

file_path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv'
cell_df = pd.read_csv(file_path)

# we want to show all columns when viewing the data
pd.set_option('display.max_columns', None)
cell_df.head()
cell_df.describe()
cell_df.info
cell_df.dtypes
cell_df.shape

# viewing data at a high lievel 
# The Class field contains the diagnosis, as confirmed by separate medical procedures, 
# as to whether the samples are benign (value = 2) or malignant (value = 4).
# Let's look at the distribution of the classes based on Clump thickness
# and Uniformity of cell size:

ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
# plt.show() - uncomment to show graph

## DATA PREPROCESSING

cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes

# using selected features

# Defining numpy array of predictors
X = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']].values
# X[0:5]

# defining target variable
y = cell_df["Class"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
print("Train test shape: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

## SVM

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 

preds = clf.predict(X_test)

# print (preds [0:5])
# print (y_test[0:5])

print("Preliminary accuracy: % .2f" % accuracy_score(y_test, preds))

# Trying for the different parameters of max_depth

# Define objective function
def objective(trial):
    # Define search space for max_depth k
    k = trial.suggest_categorical('kernel', ['sigmoid', 'rbf', 'linear', 'poly']) # searching different kernels
       
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

    # Train DT model
    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train,y_train)

    # Evaluate model
    preds = clf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    return accuracy

# Set up Optuna study
study = optuna.create_study(direction='maximize')  # maximize accuracy
study.optimize(objective, n_trials=100)  # number of trials for optimization

# Get the best hyperparameters
best_params = study.best_params
best_accuracy = study.best_value

print("Best Parameters:", best_params)
print("Best Accuracy:", best_accuracy)

# use best params for prediction
clf=svm.SVC(kernel=best_params['kernel'])
clf.fit(X_train,y_train)

# Predicting on the test set
yhat = clf.predict(X_test)
# yhat[0:5]

print("Train set accuracy: .%2f" % accuracy_score(y_train,clf.predict(X_train)))
print("Test set accuracy: %.2f" % accuracy_score(y_test, yhat))

# confusion matrix on the test set
cm = confusion_matrix(y_test, yhat)
# classification report on the test set 
cr = classification_report(y_test, yhat)
# print(cm)
# print(cr)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Predicted vs. True Cell Type on the Test Set")
plt.savefig("SVMmodelCM.png")
plt.close()
print("Confusion Matrix of the Model saved")
# plt.show()


