from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
import itertools
import numpy as np

diabetes = load_diabetes()
rf = RandomForestRegressor(n_estimators=10).fit(diabetes.data, diabetes.target)

from sklearn.inspection import partial_dependence
univariate = {}
for i in range(diabetes.data.shape[1]):
    univariate[i] = partial_dependence(rf, diabetes.data, features=[i,i+1,i+2,i+3], kind='average')['average']
    
bivariate = {}
for i, j in itertools.combinations(range(diabetes.data.shape[1]), 2):
    bivariate[(i, j)] = partial_dependence(rf, diabetes.data, features=[i, j], kind='average')['average']

h = np.zeros((diabetes.data.shape[1], diabetes.data.shape[1]))
for i, j in itertools.combinations(range(diabetes.data.shape[1]), 2):
    h[i, j] = ((bivariate[(i, j)] - univariate[i].reshape(1, -1, 1) - univariate[j].reshape(1, 1, -1) + diabetes.target.mean() ) ** 2).sum() / ((bivariate[(i, j)] - diabetes.target.mean())** 2).sum()