import numpy as np
from scipy.io import arff
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.model_selection import cross_val_score

data, meta = arff.loadarff("diabetes.arff")
X = np.empty((0, 8), np.float)
y = np.empty((0, 1), np.str)

for e in data:
    e2 = list(e)
    X = np.append(X, [e2[0:8]], axis=0)
    y = np.append(y, e2[8:9])

clf1 = BaggingClassifier()
scores = cross_val_score(clf1, X, y, cv=10)
print("_" * 45)
print("BaggingClassifier__1")
print("{0:4.2f} +/- {1:4.2f} %".format(scores.mean() * 100, scores.std() * 100))

clf2 = RandomForestClassifier()
scores = cross_val_score(clf2, X, y, cv=10)
print("_" * 45)
print("RandomForestClassifier__2")
print("{0:4.2f} +/- {1:4.2f} %".format(scores.mean() * 100, scores.std() * 100))

clf3 = AdaBoostClassifier()
scores = cross_val_score(clf3, X, y, cv=10)
print("_" * 45)
print("AdaBoostClassifier__3")
print("{0:4.2f} +/- {1:4.2f} %".format(scores.mean() * 100, scores.std() * 100))

clf4 = GradientBoostingClassifier()
scores = cross_val_score(clf4, X, y, cv=10)
print("_" * 45)
print("GradientBoostingClassifier__4")
print("{0:4.2f} +/- {1:4.2f} %".format(scores.mean() * 100, scores.std() * 100))
"""
_____________________________________________
BaggingClassifier__1
72.78 +/- 5.54 %
_____________________________________________
RandomForestClassifier__2
76.83 +/- 4.89 %
_____________________________________________
AdaBoostClassifier__3
75.52 +/- 5.71 %
_____________________________________________
GradientBoostingClassifier__4
76.17 +/- 5.05 %"""
