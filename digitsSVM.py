from sklearn import svm, datasets

# 数字のデータセット
digits = datasets.load_digits()

# 最後から10件を除く学習モデル作成
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-10], digits.target[:-10])

# 最後の10件でラベルを推定
print(clf.predict(digits.data[-10:]))
print(digits.target[-10:])
