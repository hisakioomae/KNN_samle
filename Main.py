# -*- coding: utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
import numpy as np
import csv


# CSVから特徴量読み込み
# K近傍法の実装
# 交差検証法にはLOOCV(Leave-One-Out Cross Validation)
def main():
    K = 3  # K-NNのKの値
    correct_answer_count = 0  # 正解数初期化
    x = []  # 初期化
    y = []  # 初期化
    f = open("data/data_a.csv", 'r')  # fileのopen
    data = csv.reader(f)  # ファイルの読み込み(str型)

    for row in data:
        x1 = []
        y.append(int(row[0]))  # ラベルを取得
        for i in range(1, len(row)):  # 1列目から最後の列まで
            x1.append(float(row[i]))
        x.append(x1)

    x = np.array(x)  # xをnumpy行列に変換
    y = np.array(y)  # yをnumpy行列に変換

    loo = LeaveOneOut()  # LOOCVのインスタンス生成

    entire_count = loo.get_n_splits(x)  # テスト回数取得(csvファイルの行数)

    neigh = KNeighborsClassifier(n_neighbors=K)  # K-NNのインスタンス生成

    for train_index, test_index in loo.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        neigh.fit(x_train, y_train)
        result = neigh.predict(x_test)
        if result == y_test:  # 出力したラベルと元々のラベルが一致していれば
            correct_answer_count += 1

    rate = (float(correct_answer_count) / float(entire_count))  # 正解率を計算
    print(str(rate))  # 正解率を出力


if __name__ == '__main__':
    main()
