# coding: UTF-8
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

# MNISTのテストデータをダウンロード
train, test = datasets.get_mnist()
print(type(train))
print(train[0])
print(type(test))
print(test[0])

# イテレーターのセット？
# 学習用は、試行回ごとにシャッフルする
# テスト用は、シャッフルする必要はない
train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

class MLP(Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            l1 = L.Linear(None, n_units),
            l2 = L.Linear(None, n_units),
            l3 = L.Linear(None, n_out)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

model = L.Classifier(MLP(100, 10))
serializers.load_npz('mnist.model', model)

# Show the output
x, t = test[1]
print('label:', t)

print(type(x))
print(x.shape)
x = x[None, ...]
print(type(x))
print(x.shape)

y = model.predictor(x)
print('this is y:', y)
print('this is y.data:', y.data)
print('this is y.data.argmax(axis=1):', y.data.argmax(axis=1))

print('predicted_label:', y.data.argmax(axis=1)[0])
