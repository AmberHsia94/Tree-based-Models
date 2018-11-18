# coding=utf-8
import numpy as np

'''
Implementation of basic perceptron algorithm

@ peusedo-code

* 输入：T={(x1,y1),(x2,y2)...(xN,yN)}（其中xi∈X=Rn，yi∈Y={-1, +1}，i=1,2...N，学习速率为η）
输出：w, b;  感知机模型f(x)=sign(w·x+b)
(1) 初始化w0,b0，权值可以初始化为0或一个很小的随机数
(2) 在训练数据集中随机选取（x_i, y_i）
(3) 如果yi(w xi+b)≤0
		   w = w + η y_i x_i
		   b = b + η y_i
(4) 转至（2）,直至训练集中没有误分类点
--------
@:param

features: the input feature dimension 
iter: the number of training iterations
lr: learning rate
x

@ author: Amber Hsia
@ 11.09.2018
'''

import pandas as pd
import numpy as np
import time
import random
# from sklearn.metrics import accuracy_score

class Perceptron(object):

	def __init__(self, train_x, train_y, iters, lr):
		self.train_x = train_x
		self.train_y = train_y
		self.train_samples = train_x.shape[0]
		self.features = train_x.shape[1]
		self.iters = iters
		self.lr = lr


	def forward(self, x):
		return np.dot(self.w, np.transpose(x))  # wx^T

	def update(self, y, x):
		return self.w+np.multiply(self.lr*y, x)  # w = w + η y_i x_i


	def train(self):
		self.w =[random.random() for i in range(self.features+1)]

		# starts training iterations
		for iter in range(self.iters):

			# choose a data point randomly
			idx = random.randint(0, self.train_samples)
			data_x = list(train_x[idx])
			data_x.append(1.0)  # bias term  --> [1, features+1]
			data_y = train_y[idx]   # 1

			if self.forward(data_x) * data_y <= 0:
				self.w = self.update(data_y, data_x)

	def predict(self, test_x):
		labels = []
		for features in test_x:
			x = list(features)
			x.append(1)
			labels.append(np.sign(self.forward(x)))  # y = sign(wx+b)
		return labels


if __name__ == '__main__':

	time_1 = time.time()

	# reading data
	df = pd.read_csv('/Users/amberhsia/Documents/algorithm/ml-basics/q4credit.csv',sep=',')
	del df[df.columns[0]]
	target = df.columns[0]

	train =  df[df['flag'] == 0].dropna(how='any').reset_index(drop=True)
	test = df[df['flag'] == 1].dropna(how='any').reset_index(drop=True)
	del train['flag'], test['flag']

	train_y = train[target].values
	test_y = test[target].values
	del train[target], test[target]

	train_x = train.values
	test_x = test.values
	print len(train_x), len(train_y), len(test_x), len(test_y)

	time_2 = time.time()
	print('Finish data preprocessing: ', time_2-time_1)

	# initialize parameters
	iters = 1000
	lr = 0.001


	p = Perceptron(train_x, train_y, iters, lr)
	p.train()
	time_3 = time.time()
	print('Finish TRAINING... ', time_3-time_2)

	y_pred = p.predict(test_x)

	score = accuracy_score(test_y, y_pred)
	print("The accruacy socre is ", score)



