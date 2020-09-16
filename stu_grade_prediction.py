# -*- coding: utf-8 -*-
"""

@author: Emmanuel
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import warnings
import time
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, OrthogonalMatchingPursuitCV, Ridge
import joblib
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# function to load and run saved models
# To easily run saved models

# function to run our regression models


def run_reg_models(regressor_names, regressors, X_train, X_test, y_train, y_test):
    counter = 0
    for name, clf in zip(regressor_names, regressors):
        result = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        model_performance = pd.DataFrame(data=[r2_score(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred))],
                                         index=["R2", "RMSE"])
        print(name + ' performance: ')
        print(model_performance)


df_raw = pd.read_csv(
    r'C:/Users/Emmanuel/Documents/Projects/Python/Student Grade Prediction/Dataset/student.csv')

# check for null values
print(df_raw.isnull().sum())

# There are only 1 empty values per each column
# so the best way to deal with the empty values
# is to simply drop all the null columns

df_raw.dropna(inplace=True)

# our target column is the G3 column which
# before we handle our categorical data
# we need find elements most correlated with G3
sns.heatmap(df_raw.corr(), annot=True)
plt.show()
# now we handle the categorical data
# from our heat map we can see that the most correlation with G3 is with:
# G1, G2, failures and Medu


def handle_cat_data(cat_feats, data):
    for f in cat_feats:
        to_add = pd.get_dummies(data[f], prefix=f, drop_first=True)
        merged_list = data.join(
            to_add, how='left', lsuffix='_left', rsuffix='_right')
        data = merged_list

    # then drop the categorical features
    data.drop(cat_feats, axis=1, inplace=True)

    return data
#----------- End of Handle cat data function ------------#


student_df = df_raw[['G1', 'G2', 'failures', 'Medu']]

cat_features = ['Medu']
# handle_cat_data(student_df, cat_features)

# divide dataset into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(
    student_df, df_raw['G3'], test_size=0.2, random_state=0)

regressor_names = ['Linear Regression', 'Ridge Regression', 'Lasso Regression',
                   'Elastic Net Regression', 'Orthongonal Matching Pursuit CV']

regressors = [
    LinearRegression(normalize=True),
    Ridge(alpha=0, normalize=True),
    Lasso(alpha=0.01, normalize=True),
    ElasticNet(random_state=0),
    OrthogonalMatchingPursuitCV(cv=8, normalize=True)
]

run_reg_models(regressor_names, regressors, X_train, X_test, y_train, y_test)


# Predict the score of one student from our dataset
# print(student_df.head(1))

# print(df_raw.head(1))
sel_reg = regressors[4]
predicted_val = sel_reg.predict(student_df.head(1))
print('Predicted Final Grade:' + str(round(predicted_val[0])))
print('Actual Final Grade: ' + str(df_raw.head(1)['G3'].values[0]))
