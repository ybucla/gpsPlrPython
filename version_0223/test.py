# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:24:59 2017

@author: ybwang
"""

from sklearn import datasets
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

boston = datasets.load_boston()
X = boston.data
y = boston.target
y[y <= y.mean()] = 0
y[y > 0] = 1

# pentalty logistic regression using given C
C = 1
clf_l1_LR = LogisticRegression(C=C, solver='liblinear',penalty='l1')
clf_l1_LR.fit(X, y)
coef_l1_LR = clf_l1_LR.coef_.ravel()
coef = np.concatenate((coef_l1_LR,clf_l1_LR.intercept_))
X_intercept = np.hstack((X,np.ones((len(X),1))))
v = np.dot(X_intercept,coef)
prob = 1 / (1+np.exp(-v))



cv_l1_LR = LogisticRegressionCV(cv=5,solver='liblinear',refit=True)
cv_l1_LR.fit(X,y)
C = cv_l1_LR.C_[0]
cv_l1_LR.predict_proba(X)
score = np.dot(X,cv_l1_LR.coef_.ravel()) + cv_l1_LR.intercept_
prob = 1 / (1 + np.exp(-score))


from sklearn.datasets import make_classification
x, y = make_classification(n_samples=10000, random_state=133)

# 直接用惩罚系数C=0.01拟合
lr = LogisticRegression(penalty='l1', solver='liblinear', C=0.01)
lr.fit(x, y)

# cv训练出最优惩罚系数为C=0.01，并用训练好的C重新拟合refit
lr_cv = LogisticRegressionCV(penalty='l1', solver='liblinear', Cs=[0.01,0.001],
refit=True)
lr_cv.fit(x, y)

# 惩罚系数一样，非0个数一样
assert lr.C == lr_cv.C_
assert np.count_nonzero(lr.coef_) == np.count_nonzero(lr_cv.coef_)

# coef非常接近 （）注意确实不一样
print lr.coef_
print lr_cv.coef_

# 手动检验predict原始数据结果
score = np.dot(x,lr_cv.coef_.ravel())+lr_cv.intercept_
prob = 1 / (1 + np.exp(-score))
predict_prob = lr_cv.predict_proba(x)[:,1]
# predict_prob == prob




C, tol = 1, 0.01
kf = KFold(n_splits=2)
for train_index, test_index in kf.split(X):
#    tprint("TRAIN:", train_index,train_index.size, "TEST:", test_index, test_index.size)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01)
    clf_l1_LR.fit(X_train, y_train)
    coef_l1_LR = clf_l1_LR.coef_.ravel()
    print coef_l1_LR


