# coding=utf-8
import numpy as np
from csv import reader
import random
from random import seed

'''
Random Forest Implementation:

sub-samples + sub-features for building the tree
and uses averaging to improve the predictive accuracy and control over-fitting

Parameters
----------
dataset: the input data
k_fold: k-fold cross validation
max_depth: the maximum depth growing a tree
min_size: the minimal number of samples in the node
sample_size: the dataset size randomly chosen building a tree
n_trees: the number of tree in a forest
n_features: the number of features used building a tree
target: the label index in the dataset

@ author: Amber Hsia
@ 09.11.2018

'''


# Load a CSV file and convert into array
def load_csv(filename):
	print('Start loading dataset....')
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)  # object that iterates over line
		for row in csv_reader:
			if not row:  # row=0
				continue
			dataset.append(row)
	return dataset


# cross validation
def cross_validation(input, k_fold):
	if k_fold <= 0:
		print('The value of k_fold must larger than 0!')
	else:
		fold_size = int(len(input)/k_fold)   # fixed
		dataset = list()

		# 支持重采样
		for i in range(k_fold):  # for each batch
			batch = list()
			while len(batch) <= fold_size:
				idx = random.randint(0, len(input)-1)	# generate random index for a batch
				batch.append(input[idx])
				if i == k_fold:  	# last batch
					continue
			dataset.append(batch)
		# print('check batch size: ', len(dataset), len(dataset[1])) # (5, 29879, 8)
		return dataset


# randomly select sub-samples for training a tree, 支持重采样
def subsample(dataset, sample_ratio):
	sample_size = round(sample_ratio * len(dataset))
	subset = list()

	for i in range(len(dataset)): # each row
		idx = random.randint(0, len(dataset)-1)
		subset.append(dataset[idx])
		if i == sample_size:
			break
	return subset

# randomly select sub-features for training a tree, 不支持重采样
def subfeature(dataset, n_features):
	feature_list = list()
	print n_features, len(dataset)
	for i in range(n_features):
		idx = random.randint(2, len(dataset[0]))  # 除去index和target项
		if idx not in feature_list:   # 不支持重复采样
			feature_list.append(idx)
	print('selecting feature idx: ', feature_list)
	return feature_list

# split samples into left-subtree and right-subtree based on chosen feature and feature value
def split_groups(dataset, split_value, fea_idx):
	# print('split groups...')
	left, right = list(), list()
	for row in dataset:
		if row[fea_idx] < split_value:
			left.append(row)
		else:
			right.append(row)
	print('groups after splitting: ', len(left), len(right))
	return left, right

# compute gini index
def compute_gini(groups, label_values, target):
	# print('compute gini....')
	# the probability of same label p_k
	gini_idx = 0.

	for group in groups:  # loop over left/right
		if len(group) == 0:
			break
		score = 0.
		for label in label_values:  # loop over label classes, 不仅仅是二分类问题
			p = [row[target] for row in group].count(label)/float(len(group))
			score += p*p  # 对每一个split的左树/右树，累加每个class的p平方
		gini = 1-score # for each group
		gini_idx += gini * (float(len(group))/sum([len(group) for group in groups]))   # all samples in the group
	return gini_idx



# compute the distribution in the terminal node and return the leaf prediction
def to_terminal(group, target=1):
	print('ENtering t0_treminal......')
	outcomes = [row[target] for row in group]  # 叶子结点的label分布
	print('the distribution of leaf node prediction: ', len(outcomes))
	return max(set(outcomes), key=outcomes.count)   # 返回叶子结点最多的class为叶子结点的预测


# select the best feature to split
def best_split(dataset, feature_idx, target):
	print('Here goes best_split function....')
	label_values = list(set(row[target] for row in dataset))
	# print('label values: ', label_values)

	b_index, b_value, b_score, b_groups = 999, 999, 999, None

	# loop over all features and select one to split
	for fea_idx in feature_idx:    # loop over 所有特征
		print('selected feature index: ', fea_idx)
		feature_values = set([row[fea_idx] for row in dataset])
		print('the unique of feature values: ', feature_values)
		for feature_value in feature_values:
		# for row in dataset:			# loop over 所有特征的取值
			groups = split_groups(dataset, feature_value, fea_idx)
			#groups = split_groups(dataset, row[fea_idx], fea_idx)
			gini = compute_gini(groups, label_values, target)  # 计算gini-index数值

			if gini < b_score:
				# b_index, b_value, b_score, b_groups = fea_idx, row[fea_idx], gini, groups
				b_index, b_value, b_score, b_groups = fea_idx, feature_value, gini, groups
	print('Here ends best_split function....')
	return {'index': b_index, 'value': b_value, 'score':b_score, 'groups': b_groups}


# Splitting node
def split(node, max_depth, min_size, features_idx, depth=1, target=1):
	print('Here goes SPLIT function....')
	left, right = node['groups']
	del node['groups']

	if not left or not right:  # check for a no split
		# print('DEBUGGING: ', len(left), len(right), len(left+right))
		node['left'] = node['right'] = to_terminal(left + right)   # make label prediction
		return

	if depth >= max_depth:   # check for max depth
		node['left'], node['right'] = to_terminal(left), to_terminal(right)  # make label prediction
		return

	# process left child
	if len(left) <= min_size:  # check the number of samples in the node
		node['left'] = to_terminal(left)
	else:
		node['left'] = best_split(left, features_idx)  # choose the best feature to split
		split(node['left'], max_depth, min_size, n_features, depth+1)   # split the node

	# process right child
	if len(right) <= min_size:  # check the number of samples in the node
		node['right'] = to_terminal(right)
	else:
		node['right'] = best_split(right, features_idx)  # choose the best feature to split
		split(node['right'], max_depth, min_size, n_features, depth+1)



# Decision Tree
def build_tree(trainset,  max_depth, min_size, features_idx, target):
	print('BUILDing Tree.....')
	# choose split
	root = best_split(trainset, features_idx, target)
	split(root, max_depth, min_size, n_features)
	return root

# Random Forest model here
# n_features: number of features used for modeling

def random_forest(trainset, testset, max_depth, min_size, sample_ratio, n_trees, n_features, target):
	trees = list()
	for i in range(n_trees):
		# randomly select sub-samples
		samples = subsample(trainset, sample_ratio)
		features_idx = subfeature(trainset, n_features)

		tree = build_tree(samples, max_depth, min_size, features_idx, target)
		print('Tree initialized here....')
		print(tree)
		trees.append(tree)

	pred = [bagging_pred(trees, row) for row in testset]
	return pred


def bagging_pred(trees, row):
	pred = [predict(tree, row) for tree in trees]
	return max(set(pred), key=pred.count)  # 返回最多pred的值


def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def evaluate(dataset, k_fold, *args):
	folds = cross_validation(dataset, k_fold)
	print('split the dataset into the following folds: ', len(folds))

	scores = list()

	for i in range(len(folds)):
		train = list(folds)
		test = train.pop(i)
		# train = train.remove(i)   # remove by value, pop by index
		# print('length of training batch after removing the testing batch: ', len(train))
		train = sum(train, [])   # merge other batches into a list
		# print('length of training batch: ', len(train))

		# model
		pred = random_forest(train, test, *args)
		actual = [row[target] for row in folds[i]]

		# evaluate
		accuracy = accuracy_metric(actual, pred)
		scores.append(accuracy)
	return scores


# Test the random forest algorithm
seed(2)

# load and prepare data
filename = 'q4credit.csv'
file = load_csv(filename)  # array
file = np.delete(np.array(file), (0), axis=0)  # delete the index row
file = np.delete(file, (7), axis=1)  # delete the last column 'flag'
# print file[:2]

k_fold = 2
max_depth = 5
min_size = 3
sample_size = 1.0
n_features = int(np.log2(len(file[0])))
n_trees = 3
target = 1


scores = evaluate(file, k_fold,max_depth, min_size, sample_size, n_trees, n_features, target)
print('Trees: %d' % n_trees)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

#
# Trees: 1
# Scores: [93.16563135910893, 93.31021741458713]
# Mean Accuracy: 93.238%

