# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def test_error(res, test):
    ys = test['y']
    yhat = res.predict(test).round()
    wrong = ((ys - yhat) ** 2).sum()
    return wrong / test.shape[0]


def fit_model(df):
    formula = "y ~ {}".format(' + '.join([x for x in df.columns[1:]]))
    mod = sm.Logit.from_formula(formula, data=df)
    res = mod.fit()
    return res


def draw(Ni=50):
    mu0 = np.zeros(10)
    mu1 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    d0 = stats.norm(mu0)
    d1 = stats.norm(mu1)

    rv = np.vstack([d0.rvs((Ni, 10)),
                    d1.rvs((Ni, 10))])
    y = np.hstack([np.zeros(Ni),
                   np.ones(Ni)]).reshape(-1, 1)

    df = pd.DataFrame(np.hstack([y, rv]))
    df = df.rename(columns=lambda x: 'X' + str(x))
    df = df.rename(columns={'X0': 'y'})
    return df
