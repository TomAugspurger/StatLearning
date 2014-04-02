# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import statsmodels.api as sm

x = pd.read_csv('../data/10x.csv', index_col=0)
y = pd.read_csv('../data/10y.csv', index_col=0)
xtest = pd.read_csv('../data/10xtest.csv', index_col=0)
ytest = pd.read_csv('../data/10ytest.csv', index_col=0)

X = pd.concat([x, xtest], ignore_index=True).values
y = y.values

clf = PCA()
PCA.fit(X, y)

R = clf.explained_variance_[:5].sum() / clf.explained_variance_.sum()
print("The first 5 components explain {0:.2f}% of the variance.".format(R * 100))

subX = clf.components_[:5].T

mod = sm.OLS(y, subX)


# 10.R.3

mod = sm.OLS(y, x.values)
res = mod.fit()
yhat = res.predict(xtest)

np.mean((ytest.values - yhat) ** 2)
