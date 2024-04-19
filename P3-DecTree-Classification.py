# This lab demonstrates use of simple decision tree using the drug dataset. 

# import standard libraries
import pandas as pd
import numpy as np
import csv
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import warnings

# import sklearn libraries
from sklearn.pipeline import Pipeline # just included for future use, can comment out. 
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, \
        confusion_matrix, classification_report, ConfusionMatrixDisplay # for easier coding 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # for decision trees
from sklearn.tree import export_graphviz # for decision trees visualization
import graphviz
import optuna # for hyperparameter optimization

warnings.filterwarnings('ignore')

## GET DATASET AND PREVIEW

file_path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
df = pd.read_csv(file_path, delimiter=",")

# we want to show all columns when viewing the data
pd.set_option('display.max_columns', None)
df.head()
df.describe()
df.info
df.dtypes
df.shape

## DATA PREPROCESSING

# encoding categorical predictors as ordinal measures (numeric)
# can use pd.to_dummies if the categories are nominal. 
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['BP'] = label_encoder.fit_transform(df['BP'])
df['Cholesterol'] = label_encoder.fit_transform(df['Cholesterol'])

# Defining numpy array of predictors
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
# X[0:5]

# defining target variable
y = df["Drug"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
print("Shape of the train test data: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

## DECISION TREE

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters; note that we can use Optuna or Gridsearch 
        # for hyperparameter optimization

drugTree.fit(X_train,y_train)
predTree = drugTree.predict(X_test)

# print (predTree [0:5])
# print (y_test[0:5])

print("Decision Tree Accuracy score: %.2f " % accuracy_score(y_test, predTree))

# Trying for the different parameters of max_depth

# Define objective function
def objective(trial):
    # Define search space for max_depth k
    k = trial.suggest_int('max_depth', 1,10)  # Searching integer values between 1 and 10
       
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

    # Train DT model
    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = k)
    drugTree.fit(X_train,y_train)

    # Evaluate model
    predTree = drugTree.predict(X_test)
    accuracy = accuracy_score(y_test, predTree)

    return accuracy

# Set up Optuna study
study = optuna.create_study(direction='maximize')  # maximize accuracy
study.optimize(objective, n_trials=100, show_progress_bar=False)  # number of trials for optimization

# Get the best hyperparameters
best_params = study.best_params
best_accuracy = study.best_value

print("Best Hyperparameters:", best_params)
print("Best Accuracy:", best_accuracy)

# use best params for prediction
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth =  best_params['max_depth'])
drugTree.fit(X_train,y_train)

# Predicting on the test set
yhat = drugTree.predict(X_test)
yhat[0:5]

print("Train set accuracy using best hyperparameters: %.2f" % accuracy_score(y_train, drugTree.predict(X_train)))
print("Test set accuracy using best hyperparameters: %.2f" % accuracy_score(y_test, yhat))

# confusion matrix on the test set
cm = confusion_matrix(y_test, yhat)
# classification report on the test set 
cr = classification_report(y_test, yhat)
# print(cm)
# print(cr)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Predicted vs. True Drug Category on the Test Set")
plt.savefig("predvsactualCM.png")
plt.close()
print("Confusion Matrix of the Model saved")
# plt.show()



