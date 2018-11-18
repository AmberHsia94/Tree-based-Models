# coding=utf-8

'''
The implementation of KNN algorithm.

（1）收集数据：确定训练样本集合测试数据；
（2）计算测试数据和训练样本集中每个样本数据的距离；
（3）按照距离递增的顺序排序；
（4）选取距离最近的k个点；
（5）确定这k个点中分类信息的频率；
（6）返回前k个点中出现频率最高的分类，作为当前测试数据的分类。

@Author: AmberHsia
@Date: 18.11.2018

'''

import time
import pandas as pd
import numpy as np
import operator
from sklearn.metrics import accuracy_score


class knn(object):

	def __init__(self, train_x, train_y, test_x, test_y, k, dist):
		self.train_x = train_x
		self.train_y = train_y
		self.test_x = test_x
		self.test_y = test_y
		self.k = k
		self.dist = dist

	def distance(self, train, test):
		# print('compute distance: ')
		# print train
		# print test
		if self.dist == 'manhattan':
			return np.sum(np.sign(train - test))
		elif self.dist == 'euclidean':
			return np.sqrt(np.sum(pow(np.sign(train-test), 2)))


	def predict(self):
		dis = {}
		y_pred = []

		for test_row in range(len(self.test_x)):  # loop over each samples
			for train_row in range(len(self.train_x)):
				# compute distance between all training samples
				dis[train_row] = self.distance(self.train_x[train_row], self.test_x[test_row])

			# find the closest k neighbors and it index
			sorted_x = sorted(dis.items(), key=operator.itemgetter(1))[:k] # from smallest to largest
			# print('original dis: ', dis)
			# print('sort: ', sorted_x)

			# k-neighbors
			index = [i[0] for i in sorted_x]
			label = [train_y[idx] for idx in index]
			# print index
			# print label

			# major votes
			votes = {}
			for i in label:
				if i in votes:
					votes[i] += 1
				else:
					votes[i] = 1
			item = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)[0]
			label = item[0]
			y_pred.append(label)

		return y_pred


if __name__ == '__main__':
	time_1 = time.time()

	# reading data
	df = pd.read_csv('/Users/amberhsia/Documents/algorithm/ml-basics/q4credit.csv', sep=',')
	del df[df.columns[0]]
	target = df.columns[0]

	train = df[df['flag'] == 0].dropna(how='any').reset_index(drop=True)
	test = df[df['flag'] == 1].dropna(how='any').reset_index(drop=True)
	del train['flag'], test['flag']

	train_y = train[target].values
	test_y = test[target].values
	del train[target], test[target]

	train_x = train.values
	test_x = test.values
	print len(train_x), len(train_y), len(test_x), len(test_y)
	print train_x.shape

	time_2 = time.time()
	print('Finish data preprocessing: ', time_2 - time_1)

	# initialize parameters
	k = 5
	dist = 'euclidean'

	p = knn(train_x,train_y, test_x, test_y, k, dist)
	pred = p.predict()

	print('predict length: ',len(pred))

	score = accuracy_score(test_y, pred)
	print("The accruacy socre is ", score)


