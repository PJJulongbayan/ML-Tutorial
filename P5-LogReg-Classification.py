# This lab demonstrates use of simple logistic regression using the customer churn dataset. 

# import standard libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import warnings

# import sklearn libraries
from sklearn.pipeline import Pipeline # just included for future use, can comment out. 
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, \
        confusion_matrix, classification_report, ConfusionMatrixDisplay, log_loss # for easier coding 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import optuna # for hyperparameter optimization
from sklearn.linear_model import LogisticRegression # for logistic regression

warnings.filterwarnings('ignore')

## GET DATASET AND PREVIEW

file_path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv'
df = pd.read_csv(file_path)

# we want to show all columns when viewing the data
pd.set_option('display.max_columns', None)
df.head()
df.describe()
df.info
df.dtypes
df.shape

## DATA PREPROCESSING

# using selected features
churn_df = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
churn_df['churn'] = df['churn'].astype('int') # requirement for scikit-learn
churn_df.head()

# Defining numpy array of predictors
X = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']].values
# X[0:5]

# defining target variable
y = churn_df["churn"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
print("Test train data shape: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

## LOGISTIC REGRESSION

LR = LogisticRegression(C=0.01, solver='liblinear') # used liblinear since dataset is small
LR.fit(X_train,y_train)

preds = LR.predict(X_test)

# print (preds [0:5])
# print (y_test[0:5])

print("Preliminary accuracy %.2f" % accuracy_score(y_test, preds))

# Trying for the different parameters of C

# Define objective function
def objective(trial):
    # Define search space for max_depth k
    k = trial.suggest_float('C', 0.01,0.05)  # Searching float values between 1 and 10
       
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

    # Train DT model
    LR = LogisticRegression(C=k, solver='liblinear') 
    LR.fit(X_train,y_train)

    # Evaluate model
    preds = LR.predict(X_test)
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
LR=LogisticRegression(C=best_params['C'], solver='liblinear')
LR.fit(X_train,y_train)

# Predicting on the test set
yhat = LR.predict(X_test)
# yhat[0:5]

print("Train set accuracy: %.2f" % accuracy_score(y_train, LR.predict(X_train)))
print("Test set accuracy: %.2f" % accuracy_score(y_test, yhat))

# check logloss
yhat_prob = LR.predict_proba(X_test)
log_loss(y_test, yhat_prob)

# confusion matrix on the test set
cm = confusion_matrix(y_test, yhat)
# classification report on the test set 
cr = classification_report(y_test, yhat)
# print(cm)
# print(cr)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Predicted vs. True Customer Churn on the Test Set")
plt.savefig("LogRegmodelCM.png")
plt.close()
print("Confusion Matrix of the Model saved")
# plt.show()


