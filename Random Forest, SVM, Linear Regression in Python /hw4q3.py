## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms to detect eye state

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

######################################### Reading and Splitting the Data ###############################################
# XXX
# TODO: Read in all the data. Replace the 'xxx' with the path to the data set.
# XXX
data = pd.read_csv('eeg_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data.
random_state = 100

# XXX
# TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 100.
# XXX
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=True, random_state=100)


# ############################################### Linear Regression ###################################################
# XXX
# TODO: Create a LinearRegression classifier and train it.
# XXX
lr = LinearRegression().fit(x_train, y_train)
y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

# XXX
# TODO: Test its accuracy (on the training set) using the accuracy_score method.
# TODO: Test its accuracy (on the testing set) using the accuracy_score method.
# Note: Round the output values greater than or equal to 0.5 to 1 and those less than 0.5 to 0. You can use y_predict.round() or any other method.
# XXX

print('Train accuracy - Lin Reg : ', 100*accuracy_score(y_train, y_train_pred.round()).round(3))
print('Test accuracy - Lin Reg : ', 100*accuracy_score(y_test, y_test_pred.round()).round(3))

# ############################################### Random Forest Classifier ##############################################
# XXX
# TODO: Create a RandomForestClassifier and train it.
# XXX

rand_for = RandomForestClassifier(random_state=42)
rand_for.fit(x_train, y_train)
y_train_pred = rand_for.predict(x_train)
y_test_pred = rand_for.predict(x_test)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX

print('Train accuracy - RandomForestClassifier : ', 100*accuracy_score(y_train, y_train_pred.round()).round(3))
print('Test accuracy - RandomForestClassifier : ', 100*accuracy_score(y_test, y_test_pred.round()).round(3))

# XXX
# TODO: Determine the feature importance as evaluated by the Random Forest Classifier.
#       Sort them in the descending order and print the feature numbers. The report the most important and the least important feature.
#       Mention the features with the exact names, e.g. X11, X1, etc.
#       Hint: There is a direct function available in sklearn to achieve this. Also checkout argsort() function in Python.
# XXX

from sklearn.metrics import r2_score
from rfpimp import permutation_importances
def r2(rf, X_train, y_train):
    return r2_score(y_train, rf.predict(X_train))
perm_imp_rfpimp = permutation_importances(rand_for, x_train, y_train, r2)
print(perm_imp_rfpimp)

# rand_for.fit(x_train, y_train)
# importances = rand_for.feature_importances_
# std = np.std([tree.feature_importances_ for tree in rand_for.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]

# # Print the feature ranking
# print("Feature ranking:")
# print(indices)

# for f in range(x_train.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# XXX
# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# Get the training and test set accuracy values after hyperparameter tuning.
# XXX

param_grid = {
    'n_estimators': [10, 30, 50],
    'max_depth': [3, 8, 15]
}
CV_rfc = GridSearchCV(estimator=rand_for, param_grid=param_grid, cv= 10)
CV_rfc.fit(x_train, y_train)
print (CV_rfc.best_params_)
print (CV_rfc.best_score_)
y_test_pred_grid = CV_rfc.predict(x_test)
print('Test Accuracy - Random Forrest GridSearchCV :', 100*accuracy_score(y_test, y_test_pred_grid.round()).round(3))

# ############################################ Support Vector Machine ###################################################
# XXX
# TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
# TODO: Create a SVC classifier and train it.
# XXX

scale = StandardScaler().fit(x_train)
x_train_svm = scale.transform(x_train)
x_test_svm = scale.transform(x_test)
svm = SVC(random_state=42)
svm.fit(x_train_svm, y_train)
y_train_pred_svm = svm.predict(x_train_svm)
y_test_pred_svm = svm.predict(x_test_svm)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX
print('Train accuracy - SVM : ', 100*accuracy_score(y_train, y_train_pred_svm.round()).round(3))
print('Test accuracy - SVM : ', 100*accuracy_score(y_test, y_test_pred_svm.round()).round(3))

# XXX
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# Get the training and test set accuracy values after hyperparameter tuning.
# XXX
##________________________________________________________________
param_grid = {
     'kernel': ['linear', 'rbf'],
     'C': [1,10, 20]
 }
svm = SVC()
CV_svm = GridSearchCV(estimator=svm, param_grid=param_grid, cv= 10)
CV_svm.fit(x_train_svm, y_train)
print (CV_svm.best_params_)
print (CV_svm.best_score_)
y_test_pred_svmgrid = CV_svm.predict(x_test_svm)
print('Test Accuracy - SVM GridSearchCV :', 100*accuracy_score(y_test, y_test_pred_svmgrid.round()).round(3))

##_____________________________________________________________________________

# XXX
# TODO: Calculate the mean training score, mean testing score and mean fit time for the
# best combination of hyperparameter values that you obtained in Q3.2. The GridSearchCV
# class holds a  ‘cv_results_’ dictionary that should help you report these metrics easily.
# XXX

print("GridSearchCV Table", CV_svm.cv_results_)

# ######################################### Principal Component Analysis #################################################
# XXX
# TODO: Perform dimensionality reduction of the data using PCA.
#       Set parameters n_component to 10 and svd_solver to 'full'. Keep other parameters at their default value.
#       Print the following arrays:
#       - Percentage of variance explained by each of the selected components
#       - The singular values corresponding to each of the selected components.
# XXX

pca = PCA(n_components=10, svd_solver='full')
pca.fit(x_data)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
