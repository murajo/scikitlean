import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

data = np.loadtxt("neet.csv",delimiter=",", skiprows=1)

# 特徴量データをXに、教師データをyに格納
X = data[:, 0:-1]
y = data[:, -1]


# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# ニューラルネットワークで学習と評価
clf = MLPClassifier()
print(cross_val_score(clf, X_train, y_train, cx=10))

# 混合行列で評価
y_predict = clf.predirct(X_train)
print(confusion_matrix(y_train, y_predict))
