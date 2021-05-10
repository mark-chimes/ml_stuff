# -*- coding: utf-8 -*-

'''
Taken from https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X_new = diabetes_X[:, np.newaxis, 2]

diabetes_X_train = diabetes_X_new[:-20]
diabetes_X_test = diabetes_X_new[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

regr = linear_model.LinearRegression()

regr.fit(diabetes_X_train, diabetes_y_train)

# diabetes_y_train = regr.predict(diabetes_X_train)
diabetes_y_train_pred = regr.predict(diabetes_X_train)
diabetes_y_pred = regr.predict(diabetes_X_test)

print('Coefficients: \n', regr.coef_)
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print('Coefficient of determination: %.2f' # 1 is perfect prediction
      % r2_score(diabetes_y_test, diabetes_y_pred)) 


plt.scatter(diabetes_X_train, diabetes_y_train,  color='black')
plt.plot(diabetes_X_train, diabetes_y_train_pred, color='blue', linewidth=3)

plt.show()

plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.scatter(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

#plt.xticks(())
#plt.yticks(())

plt.show()


# plt.scatter(diabetes_X, diabetes_y,  color='black')
# plt.plot(diabetes_X, diabetes_y, color='blue', linewidth=3)