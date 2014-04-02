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

clf = PCA(n_components=5)
clf.fit(X, y)

# 10.R.1

R = clf.explained_variance_ratio_[:5].sum()
print("The first 5 components explain {0:.2f}% of the variance.".format(R * 100))

# 10.R.2
# Transform the first 300 (x) into new space on first 5 components
subX = X[:300].dot(clf.components_.T)
mod = sm.OLS(y, subX)
res = mod.fit()

print(res.mse_resid)

# 10.R.3

mod = sm.OLS(y, x.values)
res = mod.fit()
yhat = res.predict(xtest)

print(np.mean((ytest.values - yhat) ** 2))
