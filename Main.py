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
        x1 = []
        y.append(int(row[0]))  # ラベルを取得
        x1.append(float(row[1]))  # 特徴ベクトルを取得

        match_count = float(row[2])

        for i in range(3, 21):
            x1.append(float(row[i]) / match_count)
        for i in range(21, 23):
            x1.append(float(row[i]))
        x.append(x1)

    neigh.fit(x, y)
    print(x)
    print(y)

    # 予測モデルをシリアライズ
    joblib.dump(neigh, 'output/model.pkl')
    # # 予測モデルをデシリアライズ
    clf = joblib.load('output/model.pkl')

    predict_data = [0.344, 4.204379562043796, 3.562043795620438, 0.7007299270072993, 1.2262773722627738, 0.20437956204379562, 0.021897810218978103, 0.1678832116788321, 1.9781021897810218, 0.5474452554744526, 0.0948905109489051, 0.021897810218978103, 0.0072992700729927005, 0.043795620437956206, 0.5912408759124088, 0.014598540145985401, 0.0, 0.48905109489051096, 0.043795620437956206, 0.555, 0.433]
    # predict_data = [0.344, 4.204379562043796, 3.562043795620438, 0.7007299270072993, 1.2262773722627738, 0.20437956204379562, 0.021897810218978103, 0.1678832116788321, 1.9781021897810218, 0.5474452554744526, 0.0948905109489051, 0.021897810218978103, 0.0072992700729927005, 0.043795620437956206, 0.5912408759124088, 0.014598540145985401, 0.0, 0.48905109489051096, 0.043795620437956206, 0.555, 0.433]
    predict_data2 = []
    a = len(predict_data)

    predict_data2.append(predict_data)
    print(clf.predict(predict_data2))

    print(a)


if __name__ == '__main__':
    main()
