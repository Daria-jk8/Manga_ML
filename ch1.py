from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge, Lasso

boston = load_boston()

X = boston.data
y = boston.target

lr1 = LinearRegression()
lr1.fit(X, y)

print("Linear Regression")

for f, w in zip(boston.feature_names, lr1.coef_):
    print("{0:7s}: {1:6.2f}".format(f, w))
print("coef = {0:4.2f}".format(sum(lr1.coef_**2)))

""" Linear Regression
CRIM   :  -0.11
ZN     :   0.05
INDUS  :   0.02
CHAS   :   2.69
NOX    : -17.77
RM     :   3.81
AGE    :   0.00
DIS    :  -1.48
RAD    :   0.31
TAX    :  -0.01
PTRATIO:  -0.95
B      :   0.01
LSTAT  :  -0.52
coef = 340.85"""

lr2 = Ridge(alpha=10.0)
lr2.fit(X, y)
print("Ridge")

for f, w in zip(boston.feature_names, lr2.coef_):
    print("{0:7s}:{1:6.2f}".format(f, w))

print("coef = {0:4.2f}".format(sum(lr2.coef_**2)))

"""Ridge
CRIM   : -0.10
ZN     :  0.05
INDUS  : -0.04
CHAS   :  1.95
NOX    : -2.37
RM     :  3.70
AGE    : -0.01
DIS    : -1.25
RAD    :  0.28
TAX    : -0.01
PTRATIO: -0.80
B      :  0.01
LSTAT  : -0.56
coef = 25.74"""

lr3 = Lasso(alpha=2.0)
lr3.fit(X, y)
print("Lasso")

for f, w in zip(boston.feature_names, lr3.coef_):
    print("{0:7s}: {1:6.2f}".format(f, w))

print("coef = {0:4.2f}".format(sum(lr3.coef_**2)))
"""Lasso
CRIM   :  -0.02
ZN     :   0.04
INDUS  :  -0.00
CHAS   :   0.00
NOX    :  -0.00
RM     :   0.00
AGE    :   0.04
DIS    :  -0.07
RAD    :   0.17
TAX    :  -0.01
PTRATIO:  -0.56
B      :   0.01
LSTAT  :  -0.82
coef = 1.02"""
