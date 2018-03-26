# coding: UTF-8
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import pandas as pd

DEBUG_MODE = True

# Utilities ================
# FIXME 後で別のファイルにうつしてimportして使う
def d_print(msg):
    if DEBUG_MODE:
        print(msg)
def d_print_hr():
    print("=======================================")
    print("")

# ==========================
# 設定
# FIXME ここでサイズを定義できるようにする
batchsize = 100 # 確率的勾配降下法で学習させる際の１回分のバッチサイズ
n_epoch   = 20 # 学習の繰り返し回数
n_units   = 1000 # 中間層の数
N = 30 # 検証数
# ==========================


# データ取得
iris_data = pd.read_csv('./csv/iris.csv', header=None).values
d_print(type(iris_data))
d_print(iris_data[0])
d_print_hr()

setosa, versicolor, virginica = np.split(iris_data, [50, 100])

d_print(len(setosa))
d_print(setosa[-1])
d_print(len(versicolor))
d_print(versicolor[-1])
d_print(len(virginica))
d_print(virginica[-1])
d_print_hr()

# 単純化のために、setosa と versicolorだけ使用する
data_setosa = setosa
data_versicolor = versicolor

data_setosa[:,4] = 0
data_versicolor[:,4] = 1

data_setosa = data_setosa.astype('float32')
data_versicolor = data_versicolor.astype('float32')

d_print(data_setosa)
d_print(data_versicolor)
d_print_hr()

# 学習用データと検証用データに分ける
train = np.concatenate((data_setosa[0:40], data_versicolor[0:40]))
test = np.concatenate((data_setosa[40:], data_versicolor[40:]))

d_print(train[0])

# Train: 学習用データ
## ラベルデータと予測データに分ける
train_data = train[:,0:4]
train_label = train[:,4].astype('int32')

## Chainerが予測可能な形式に変換
train_dataset = chainer.datasets.TupleDataset(train_data, train_label)
d_print(train_dataset[0])

# Test: 検証用データ
## ラベルデータと予測データに分ける
test_data = test[:,0:4]
test_label = test[:,4].astype('int32')

## Chainerが予測可能な形式に変換
test_dataset = chainer.datasets.TupleDataset(train_data, train_label)

# Iterator の作成
train_iter = iterators.SerialIterator(train_dataset, batch_size=100, shuffle=True)
test_iter = iterators.SerialIterator(test_dataset, batch_size=100, repeat=False, shuffle=False)

d_print(type(train_iter))
d_print(train_iter)


# exit()

# モデルの設定

## Prepare multi-layer perceptron model
## 多層パーセプトロンモデルの設定
class MultiLayerPerceptron(Chain):
    def __init__(self, n_units, n_out):
        super(MultiLayerPerceptron, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out
            # self.l1 = L.Linear(4, 100)  # n_in -> n_units
            # self.l2 = L.Linear(100, 100)  # n_units -> n_units
            # self.l3 = L.Linear(100, 2)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

## 最初の引数はノード数
## 2つ目の引数は導く解の数
model = L.Classifier(MultiLayerPerceptron(100,2))

# optimizerの設定
# 今回はAdamを使用
# https://www.scribd.com/doc/260859670/30minutes-Adam
optimizer = optimizers.Adam()
optimizer.setup(model)

# trainerの設定
updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (20, 'epoch'), out='result')


# ログなどの便利ツール =======================

# モデルを検証用データで毎回評価する
trainer.extend(extensions.Evaluator(test_iter, model))

# 評価の値をログとして残しておく
trainer.extend(extensions.LogReport())

# 指定された項目をコマンドライン上に表示
# ここでいうmainはmain optimizerの対象となるlinkのことで、
# validationは評価用エクステンションのデフォルトの名前のこと。
# epoch 以外の項目はClassifier Linkでレポートされ、updater か evaluatorで呼び出される
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))

# 進行度合いをプログレスバーで表示
trainer.extend(extensions.ProgressBar())
# ============================================

trainer.run()
