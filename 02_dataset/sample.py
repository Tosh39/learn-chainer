# coding: UTF-8
# http://s0sem0y.hatenablog.com/entry/2016/12/20/220928
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets
from chainer.training import extensions
import numpy as np
from sklearn.datasets import load_iris
import chainer.computational_graph as c
from sklearn import preprocessing

# load data
iris = load_iris()
print(iris)

X = iris.data
X = preprocessing.scale(X)		#mean of zero and a standard deviation of one. mean = 0  variance = 1
#X = preprocessing.normalize(X)    # rescaling real valued numeric attributes into the range 0 and 1.
X = X.astype(np.float32)
Y = iris.target
Y = Y.flatten().astype(np.int32)

# prepare train-data and test-data for validation
train ,test= datasets.split_dataset_random(chainer.datasets.TupleDataset(X,Y),100)
print(train[0])
print(test[0])
train_iter = chainer.iterators.SerialIterator(train, 30)	# 30 = minibatch sizes
test_iter = chainer.iterators.SerialIterator(test, 1,repeat=False, shuffle=False)


# define 4 lyaers networks
class IrisNN(chainer.Chain):
	def __init__(self,hidden = [100,200,100]): # hidden = [number of units of 1st hidden layer, that of 2nd, that of 3rd]
		super(IrisNN,self).__init__(
                l1 = L.Linear(4,hidden[0]),
                bn1 = L.BatchNormalization(hidden[0]),
                l2 = L.Linear(hidden[0],hidden[1]),
                bn2 = L.BatchNormalization(hidden[1]),
                l3 = L.Linear(hidden[1],hidden[2]),
                bn3 = L.BatchNormalization(hidden[2]),
                l4 = L.Linear(hidden[2],3),
                bn4 = L.BatchNormalization(3))

	def __call__(self,x):
		h = self.l1(x)
#		h = self.bn1(h,test=False, finetune=False)	#Batch Normalization
		h = F.relu(h)
		h = self.l2(h)
#		h = self.bn2(h,test=False, finetune=False)
		h = F.relu(h)
		h = F.dropout(h,ratio=0.5)	# dropout 0.5 * 200 units
		h = self.l3(h)
#		h = self.bn3(h,test=False, finetune=False)
		h = F.relu(h)
		h = self.l4(h)
#		h = self.bn4(h,test=False, finetune=False)
		return h

model = L.Classifier(IrisNN())
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer, device=-1) # device=-1 mean using only CPU
trainer = training.Trainer(updater, (100, 'epoch'), out="result")   #  training 100 epochs
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())

trainer.run()
