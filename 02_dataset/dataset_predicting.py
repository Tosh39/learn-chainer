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
    if DEBUG_MODE:
        print("=======================================")
        print("")

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

data_setosa = setosa
data_versicolor = versicolor
data_virginica = virginica

data_setosa[:,4] = 0
data_versicolor[:,4] = 1
data_virginica[:,4] = 2

data_setosa = data_setosa.astype('float32')
data_versicolor = data_versicolor.astype('float32')
data_virginica = data_virginica.astype('float32')

d_print(data_setosa)
d_print(data_versicolor)
d_print(data_virginica)
d_print_hr()

# 学習用データと検証用データに分ける
train = np.concatenate((data_setosa[0:40], data_versicolor[0:40], data_virginica[0:40]))
test = np.concatenate((data_setosa[40:], data_versicolor[40:], data_virginica[40:]))

d_print(train[0])

# Train: 学習用データ
## ラベルデータ(目的変数)と予測データ(説明変数)に分ける
train_data = train[:,0:4]
train_label = train[:,4].astype('int32')

## Chainerが予測可能な形式に変換
train_dataset = chainer.datasets.TupleDataset(train_data, train_label)
d_print(train_dataset[0])

# Test: 検証用データ
## ラベルデータ(目的変数)と予測データ(説明変数)に分ける
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

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

## 最初の引数はノード数
## 2つ目の引数は導く解の数
model = L.Classifier(MultiLayerPerceptron(100,3))

# 前回の学習内容を読み込む設定
#   npzからでもスナップショットからでも同じ内容を読み込むので
#   predictorの精度に影響はない
# ## npzファイルから読み込みはこちら
# serializers.load_npz("iris_triple.npz", model)

## スナップショットから読み込みするのであればこちら
serializers.load_npz("result/snapshot", model, path="updater/model:main/")
## スナップショットファイルはネットワークの情報以外の情報を含んでいるため
## スナップショットから読み込みするのであれば
## pathの指定が必要: 'updater/model:main/'
## https://qiita.com/ka10ryu1/items/749dd61b7494adf12dc2


# Numpy.arrayしか受け付けていないっぽいので、一旦Numpy.arrayにする
# https://docs.chainer.org/en/stable/tutorial/train_loop.html
# # 直接指定する場合
# x = np.array([5.7,3.0,4.2,1.2], dtype="float32")
# d_print(x.shape)

# 引っ張って来る場合
index = 2
## このanswerのうちどれかをコメントアウト
answer = data_setosa[index]
# answer = data_versicolor[index]
# answer = data_virginica[index]

x = answer[0:4]
d_print(x.shape)

x = x[None, ...]
d_print(x.shape)

y = model.predictor(x)
print("y: ", y)
# model.predictor(x)の返り値としてそれぞれの可能性がvariableで取得できる
#  -> 例: variable([[ 8.031654 ,  2.9862404, -6.380532 ]])
# これのうち、一番大きい値のものである確率が高いという意味
# この場合だと、0に該当する setosa である可能性が高いということになる

guess = y.data.argmax(axis=1)[0]
print("y.data.argmax(axis=1)[0]: ", guess)

label = answer[4].astype(int)
print("Correct Answer: ", ['setosa', 'versicolor', 'virginica'][label])
print("Chainer Guess: ",['setosa', 'versicolor', 'virginica'][guess])
