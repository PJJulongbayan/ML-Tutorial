## This lab demonstrates use of simple linear regression models on the fuel consumption dataset. 

## IMPORTING LIBRARIES

# import standard libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings

# import sklearn libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline # just included for future use, can comment out. 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor

import joblib # for pkl save 

from mlxtend.feature_selection import SequentialFeatureSelector # for forward and backward selection

warnings.filterwarnings('ignore')

## READING AND VIEWING DATA

file_path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv'
df = pd.read_csv(file_path)

# we want to show all columns when viewing the data
pd.set_option('display.max_columns', None)
df.head()
df.describe()
df.info
df.dtypes
df.shape

# exploring some features of the data
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

## SIMPLE DATA VIZ

viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.savefig('simpleviz.png') # to save plot
plt.close() # close
print('Sample Viz has been saved as simpleviz.png')

# plotting fuel consumption against emission
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
# plt.show()
plt.close() # close

# plotting engine size against emission
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
# plt.show()
plt.close() # close

# plotting cylinders against emission
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
# plt.show()
plt.close() # close

## SIMPLE LINEAR REGRESSION

# linear regression on co2 emission and engine size
y_data = df["CO2EMISSIONS"]
x_data = df[["ENGINESIZE"]] # train set has to be df

# check regression plot first
sns.regplot(x = "ENGINESIZE", y = "CO2EMISSIONS", data = df)
plt.close() # close

# Train test split
# Note that same thing can be obtained by using train test split using the code commented below: 
# msk = np.random.rand(len(df)) < 0.8
# train = cdf[msk]
# test = cdf[~msk]
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state = 1)

sns.regplot(x = x_train, y = y_train)
plt.close() # close

lm = LinearRegression()
lm = lm.fit(x_train, y_train)

# check intercept value and coefficient
lm.intercept_
lm.coef_

# check performance on test set
preds = lm.predict(x_test)

print("Mean squared error using only engine size as feature: %.2f" % mean_squared_error(y_test, preds))
print("R-squared using only engine size as feature: %.2f" % r2_score(y_test, preds))

# Change x_data to other variables to run the same code on other features. 

## MULTIPLE LINEAR REGRESSION

# --- 1. Using engine size, cylinders, and fuel consumption as predictors. ---
y_data = df["CO2EMISSIONS"]
x_data = df[["ENGINESIZE", "FUELCONSUMPTION_COMB", "CYLINDERS"]] # train set has to be df 

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state = 1)
lm = LinearRegression()
lm = lm.fit(x_train, y_train)

# check intercept value and coefficient
lm.intercept_
lm.coef_

# check performance on test set
preds = lm.predict(x_test)

print("Mean squared error using multiple features: %.2f" % mean_squared_error(y_test, preds))
print("R-squared using multiple features: %.2f" % r2_score(y_test, preds))

# multiple linear regression achieved a higher score than any individual simple regression

# --- 2. Using fuel taype as an additional predictor ---
# check datatype first and convert make, model, vehicle class, transmission, and fuel type into numbers
# dummies_data = df[["MAKE", "MODEL", "VEHICLECLASS", "TRANSMISSION", "FUELTYPE"]] 
# dummies = pd.get_dummies(dummies_data , prefix=
#                        {'MAKE': 'MAKE', 'MODEL': 'MODEL', "VEHICLECLASS": "VC", "TRANSMISSION": "TX", "FUELTYPE": "FT"})

dummies = df[["FUELTYPE"]]
dummies = pd.get_dummies(dummies, prefix = df["FUELTYPE"].name)

# append new fuel type as part of predictors
x_data = pd.concat([x_data, dummies], axis = 1)
y_data = df["CO2EMISSIONS"] # use same y_data

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state = 1)
lm = LinearRegression()
lm = lm.fit(x_train, y_train)

# check intercept value and coefficient
lm.intercept_
lm.coef_

# check performance on test set
preds = lm.predict(x_test)

print("Mean squared error using multiple plus encoded categorical features: %.2f" % mean_squared_error(y_test, preds))
print("R-squared using multiple plus encoded categorical features: %.2f" % r2_score(y_test, preds))

# Very high prediction capability
# let's look at how each fuel type compares to c02 emissions
df_grp = df.groupby("FUELTYPE")["CO2EMISSIONS"].mean().reset_index()
sns.barplot(x = df["FUELTYPE"],  y = df["CO2EMISSIONS"])

## SGD REGRESSOR 

# the above code utilizes ordinary least square methods by calling LinearRegression(). 
# However, it will be time consuming when dealing with very large dataset. 
# Let's try and model using sgd regressor

from sklearn.linear_model import SGDRegressor

# Generate some example data if not available 
# X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# using engine size, cylinders, and fuel consumption as predictors.
y_data = df["CO2EMISSIONS"]
x_data = df[["ENGINESIZE", "FUELCONSUMPTION_COMB", "CYLINDERS"]] # train set has to be df 

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state = 1)

# standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
# fit is already applied on train and should not be applied to test to avoid leakage
x_test_scaled = scaler.transform(x_test)  

# Create and fit the SGDRegressor model
sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd_regressor.fit(x_train_scaled, y_train)

# check intercept value, coefficient, and other stats
sgd_regressor.intercept_
sgd_regressor.coef_
sgd_regressor.t_

# model_coef
model_coef = pd.DataFrame({'Columns': x_train.columns , 'Coefficients': sgd_regressor.coef_})

## SAVING MODEL AS PKL FILE FOR LATER USE

joblib.dump(sgd_regressor, 'sgd_regressor.pkl')
print('SGDregressor model has been saved')

## READ SAVED MODEL AND USE TO MAKE PREDICTION
with open('sgd_regressor.pkl', 'rb') as f:
    sgd_regressor = joblib.load(f)
print('SGDregressor model has been loaded')

# check performance on test set
preds = sgd_regressor.predict(x_test_scaled)

print("Mean squared error using SGDRegressor: %.2f" % mean_squared_error(y_test, preds))
print("R-squared using SGDRegressor: %.2f" % r2_score(y_test, preds))


## Forward Selection Algorithm

# using mlxtend feature selection

model = LinearRegression()
# Create a SequentialFeatureSelector object
selector = SequentialFeatureSelector(
            model,
            forward=True,  # Perform forward selection, can be changed to False for backward selection. 
            scoring='neg_mean_squared_error',  # Use mse as the evaluation metric
            cv=5)

# Fit the selector to the data
selector.fit(x_train, y_train)

# Get the selected feature indices
selected_features = selector.k_feature_idx_
selected_features_names = selector.k_feature_names_
print("Selected features:", selected_features)
print("Selected features:", selected_features_names)

## Note: Can use statsmodel instead to get statistical inference such as p-value. 