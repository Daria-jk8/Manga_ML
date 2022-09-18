from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

clf1 = LogisticRegression()
clf1.fit(X, y)

for f, w in zip(breast_cancer.feature_names, clf1.coef_[0]):
    print("{0:<23}: {1:6.2f}".format(f, w))

"""mean radius            :   0.96
mean texture           :   0.46
mean perimeter         :   0.27
mean area              :  -0.02
mean smoothness        :  -0.04
mean compactness       :  -0.17
mean concavity         :  -0.23
mean concave points    :  -0.10
mean symmetry          :  -0.05
mean fractal dimension :  -0.01
radius error           :   0.04
texture error          :   0.38
perimeter error        :   0.15
area error             :  -0.11
smoothness error       :  -0.00
compactness error      :  -0.04
concavity error        :  -0.05
concave points error   :  -0.01
symmetry error         :  -0.01
fractal dimension error:  -0.00
worst radius           :   1.02
worst texture          :  -0.50
worst perimeter        :  -0.25
worst area             :  -0.01
worst smoothness       :  -0.06
worst compactness      :  -0.52
worst concavity        :  -0.65
worst concave points   :  -0.19
worst symmetry         :  -0.16
worst fractal dimension:  -0.05"""

# do not know how to decide (?)
clf2 = DecisionTreeClassifier(max_depth=2)
clf2.fit(X, y)
