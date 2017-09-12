# -*- coding: utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import csv


def main():
    neigh = KNeighborsClassifier(n_neighbors=1)  # 最近傍法のオブジェクト生成

    f = open("data/data.csv", 'r')  # fileのopen
    data = csv.reader(f)  # ファイルの読み込み(str型)
    x = []  # 初期化
    y = []  # 初期化

    for row in data:
        x.append(row[1:])  # 特徴ベクトルを取得
        y.append(int(row[0]))  # ラベルを取得

    print(x)
    print(y)

    neigh.fit(x, y)

    # 予測モデルをシリアライズ
    joblib.dump(neigh, 'output/model.pkl')
    # 予測モデルをデシリアライズ
    clf = joblib.load('output/model.pkl')
    print(clf.predict([['0.274', '119', '471', '416', '63', '114', '22', '0', '22', '202', '68', '1', '0', '0', '3', '44', '8', '68', '13', '0.486', '0.352']]))


if __name__ == '__main__':
    main()
