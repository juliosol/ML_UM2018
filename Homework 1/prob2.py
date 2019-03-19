import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import math
import random
from random import seed
from random import randrange, shuffle
np.set_printoptions(threshold=np.inf)

random.seed(9001)

# Load dataset
dataset = datasets.load_boston()

features = dataset.data
labels = dataset.target

# Before answering the questions to the problem, we do some preprocessing
# of the data

Nsplit = 50
# Training set
X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
# Test set
X_test, y_test = features[-Nsplit:], labels[-Nsplit:]

#Normalizing the features
N = len(features[1])
Mtrain = X_train.shape[0]
Mtest = X_test.shape[0]

# Part a). This part normalizes the data set 

# This function computes the mean of a column

def Avgfinder(vec_feature):
	avg_feature = np.mean(vec_feature)
	return avg_feature

#This function computes the standard deviation of a column
def Stdevfinder(vec_feature):
		stdev = np.std(vec_feature)
		return stdev

# Function that gives a 1 by 13 array with the mean of each feature of training set

def feature_mean(vec_feature):
	prearr_featuremean = []
	for k in range(len(vec_feature)):
		meanfeature = Avgfinder(vec_feature[k])
		prearr_featuremean.append(meanfeature)
	featuremean = np.array(prearr_featuremean)
	return featuremean

# Function that gives 1 by 13 array with the stdev  of each feature of training set

def feature_stdev(vec_feature):
	prearr_feature_stdev = []
	for k in range(len(vec_feature)):
		stdev_feature = Stdevfinder(vec_feature[k])
		prearr_feature_stdev.append(stdev_feature)
	feature_stdev = np.array(prearr_feature_stdev)
	return feature_stdev

xtrain_feature_mean = feature_mean(np.transpose(X_train))
xtrain_feature_stdev = feature_stdev(np.transpose(X_train))

#print(xtrain_feature_mean)
#print(xtrain_feature_stdev)

# Function that inputs a matrix (with feature columns) and normalizes the vector 
#with respect to this mean and stdev.
def Normfeature(matrix, avg_vector, stdev_vector):
	matrix = np.transpose(matrix)
	M = matrix.shape[0] 
	N = matrix.shape[1]
	normalized_matrix_list = []
	for i in range(0,M):
		fi_mean_vector = np.ones(N)*avg_vector[i]
		if stdev_vector[i] != 0:
			row_i = (matrix[i] - fi_mean_vector)/stdev_vector[i]
			normalized_matrix_list.append(row_i)
		else:
			row_i = (matrix[i] - fi_mean_vector)
			normalized_matrix_list.append(row_i)
		normalized_matrix = np.transpose(np.array(normalized_matrix_list))
	return normalized_matrix

# Normalized data

# Normalized training set
xtrain_set_normalized = Normfeature(X_train, 
	xtrain_feature_mean,xtrain_feature_stdev)

# Normalized test set
xtest_set_normalized = Normfeature(X_test,
	xtrain_feature_mean, xtrain_feature_stdev)

#Before going to part b,c,d,e we append a bias term to the training set.

def bias_add(vec_feature):
	if len(vec_feature) == 1:
		np_bias_vec_feature = np.insert(vec_feature, 0, 1)
	else:
		bias_vec_feature = []
		for k in range(len(vec_feature)):
			vector_k = np.insert(vec_feature[k], 0, 1)
			bias_vec_feature.append(vector_k)
		np_bias_vec_feature = np.array(bias_vec_feature)
	return np_bias_vec_feature

bias_xtrain_set_normalized = bias_add(xtrain_set_normalized)
bias_xtest_set_normalized = bias_add(xtest_set_normalized)

# We use the function mse below to compute the mse. This function takes
# as input a matrix of features X, a real output vector Y and a weight vector w

def rmse(X,Y,w):
	M = X.shape[0]
	total_sum = 0
	for d in range(M):
		#yhat = np.dot(np.transpose(w),X[d])
		yhat = np.dot(np.transpose(w),X[d])
		squared_difference = np.square(yhat - Y[d])
		#print(d)
		#print(Y[d])
		#print(squared_difference)
		total_sum = total_sum + squared_difference
	#print(total_sum)
	#print(M)
	rmse_result = np.sqrt(total_sum/M)
	return rmse_result

x1_train = np.linspace(-1,500, 500)


#Part a of the problem

RMSE_training_error_veca = []

RMSE_test_error_veca = []

# Closed form w solver

def w_parameter_closed(feature_matrix,y_train):
	XTX = np.dot(np.transpose(feature_matrix),feature_matrix)
	if np.linalg.det(XTX) != 0:
		XTX_inv = np.linalg.inv(XTX)
		pseudo_inverse = np.dot(XTX_inv,np.transpose(feature_matrix))
		weight_vector = np.dot(pseudo_inverse,y_train)
	else:
		pseudo_inverse = np.linalg.pinv(feature_matrix)
		weight_vector = np.dot(pseudo_inverse,y_train) 
	return weight_vector

# Feature of order 0:

# Deriving feature 

feature0_matrix_train = np.ones((Mtrain,1))
feature0_matrix_test = np.ones((Mtest,1))
#bias_feature0_matrix = bias_add(feature0_matrix)
w0_featuredeg0 = w_parameter_closed(feature0_matrix_train,y_train)

rmse_featuredeg0_train = rmse(feature0_matrix_train,y_train,w0_featuredeg0)

RMSE_training_error_veca.append(rmse_featuredeg0_train)

print("The RMSE for training data feature of order 0 is " + 
	str(rmse_featuredeg0_train))

rmse_featuredeg0_test = rmse(feature0_matrix_test,y_test,w0_featuredeg0)

RMSE_test_error_veca.append(rmse_featuredeg0_test)

print("The RMSE for test data feature of order 0 is " 
	+ str(rmse_featuredeg0_test))

# Degree 1 feature

# Matrix of features for features of order 1
feature1_matrix_train = bias_xtrain_set_normalized
feature1_matrix_test = bias_xtest_set_normalized
#print(feature1_matrix.shape)

# Weight vector for features of order 1
w1_featuredeg1 = w_parameter_closed(feature1_matrix_train,y_train)
#print(w1_featuredeg1)

#RMSE error for feature of degree 1 using training set
rmse_featuredeg1_train = rmse(feature1_matrix_train,y_train,w1_featuredeg1)

RMSE_training_error_veca.append(rmse_featuredeg1_train)

print("The RMSE for training data feature of order 1 is " 
	+ str(rmse_featuredeg1_train))

# RMSE error for feature of degree 1 using test set

rmse_featuredeg1_test = rmse(feature1_matrix_test,y_test,w1_featuredeg1)

RMSE_test_error_veca.append(rmse_featuredeg1_test)

print("The RMSE for test data feature of order 1 is " + 
	str(rmse_featuredeg1_test))


# Degree 2 features

#Function that powers the entries in the vector to the order n, given.
def n_power_elements(feature_vector,n):
	npowered_vector = []
	L = len(feature_vector)
	for l in range(L):
		npowered_vector.append(feature_vector[l]**n)
	return npowered_vector

# We first create a functiont that appends the number 
# of the order vector into the feature matrix. THe function 
# takes as input a feature matrix that does NOT contain a bias column of ones and
# a degree n >= 2 to generate the feature matrix

def n_order_feature_matrix(feature_matrix, n):
	feature_matrix = np.transpose(feature_matrix)
	n_order_feat_matrix = feature_matrix.tolist()
	M = feature_matrix.shape[0]
	N = feature_matrix.shape[1]
	k = 2
	j = 1
	for k in range(2,n+1):
		for j in range(0,M):
			#print(j)
			n_order_feat_matrix.append(n_power_elements(feature_matrix[j],k))
	feature_matrix_array = np.transpose(np.array(n_order_feat_matrix))
	return feature_matrix_array

def n_order_feature_matrix2(feature_matrix, n):
	feature_matrix = np.transpose(feature_matrix)
	n_order_feat_matrix2 = []
	M = feature_matrix.shape[0]
	N = feature_matrix.shape[1]
	k = 2
	j = 1
	for j in range(0,M):
		n_order_feat_matrix2.append(feature_matrix[j])
		for k in range(2,n+1):
			#print(j)
			n_order_feat_matrix2.append(n_power_elements(feature_matrix[j],k))
	feature_matrix_array = np.transpose(np.array(n_order_feat_matrix2))
	return feature_matrix_array

feature2_matrix = n_order_feature_matrix2(xtrain_set_normalized,2)
feature2_matrix_test = n_order_feature_matrix2(xtest_set_normalized,2)

#Normalize feature matrix

xtrain_feature2_mean = feature_mean(np.transpose(feature2_matrix))
xtrain_feature2_stdev = feature_stdev(np.transpose(feature2_matrix))

bias_xtrain_feature2_matrix = bias_add(Normfeature(feature2_matrix,
	xtrain_feature2_mean,xtrain_feature2_stdev))

bias_xtest_feature2_matrix = bias_add(Normfeature(feature2_matrix_test,
	xtrain_feature2_mean,xtrain_feature2_stdev))


# Weight vector for features of degree 2

w2_featuredeg2 = w_parameter_closed(bias_xtrain_feature2_matrix,y_train)

# RMSE error computation for training set

rmse_featuredeg2_train = rmse(bias_xtrain_feature2_matrix,y_train,w2_featuredeg2)
RMSE_training_error_veca.append(rmse_featuredeg2_train)

#print("This is RMSE sum of deg 2" + str(total_sum(feature2_matrix, y_train, w2_featuredeg2)))

print("The RMSE for training data feature of order 2 is " 
	+ str(rmse_featuredeg2_train))

# RMSE error computation for test set
#print(y_test.shape)
#print()
rmse_featuredeg2_test = rmse(bias_xtest_feature2_matrix,y_test,w2_featuredeg2)
RMSE_test_error_veca.append(rmse_featuredeg2_test)

print("The RMSE for test data feature of order 2 is " 
	+ str(rmse_featuredeg2_test))

# Degree 3 feature

#feature3_matrix = n_order_feature_matrix2(bias_xtrain_set_normalized,3)
#print(feature3_matrix[0])

feature3_matrix = n_order_feature_matrix(xtrain_set_normalized,3)
feature3_matrix_test = n_order_feature_matrix(xtest_set_normalized,3)

#Normalize feature matrix

xtrain_feature3_mean = feature_mean(np.transpose(feature3_matrix))
xtrain_feature3_stdev = feature_stdev(np.transpose(feature3_matrix))

bias_xtrain_feature3_matrix = bias_add(Normfeature(feature3_matrix,
	xtrain_feature3_mean,xtrain_feature3_stdev))

bias_xtest_feature3_matrix = bias_add(Normfeature(feature3_matrix_test,
	xtrain_feature3_mean,xtrain_feature3_stdev))

# Weight vector for features of degree 3

w3_featuredeg3 = w_parameter_closed(feature3_matrix,y_train)

# RMSE error computation for training set

rmse_featuredeg3_train = rmse(feature3_matrix,y_train,w3_featuredeg3)
RMSE_training_error_veca.append(rmse_featuredeg3_train)

print(rmse_featuredeg3_train)

#print(y_train[50])

#print(w3_featuredeg3.shape)
# RMSE error computation for test set

rmse_featuredeg3_test = rmse(feature3_matrix_test,y_test,w3_featuredeg3)
RMSE_test_error_veca.append(rmse_featuredeg3_test)

print(rmse_featuredeg3_test)


# Degree 4 feature

feature4_matrix = n_order_feature_matrix(xtrain_set_normalized,4)
feature4_matrix_test = n_order_feature_matrix(xtest_set_normalized,4)

#Normalize feature matrix

xtrain_feature4_mean = feature_mean(np.transpose(feature4_matrix))
xtrain_feature4_stdev = feature_stdev(np.transpose(feature4_matrix))

bias_xtrain_feature4_matrix = bias_add(Normfeature(feature4_matrix,
	xtrain_feature4_mean,xtrain_feature4_stdev))

bias_xtest_feature4_matrix = bias_add(Normfeature(feature4_matrix_test,
	xtrain_feature4_mean,xtrain_feature4_stdev))

# Weight vector for features of degree 4
w4_featuredeg4 = w_parameter_closed(feature4_matrix,y_train)
#print(w4_featuredeg4)

# RMSE error computation for training set

rmse_featuredeg4_train = rmse(feature4_matrix,y_train,w4_featuredeg4)
RMSE_training_error_veca.append(rmse_featuredeg4_train)

print(rmse_featuredeg4_train)

# RMSE error computation for test set

rmse_featuredeg4_test = rmse(feature4_matrix_test,y_test,w4_featuredeg4)
RMSE_test_error_veca.append(rmse_featuredeg4_test)

print(rmse_featuredeg4_test)

x1 = np.linspace(0,4,len(RMSE_test_error_veca))
plt.plot(x1,RMSE_test_error_veca,'b',label = "Test")
plt.plot(x1,RMSE_training_error_veca,'r',label = 'Training')
plt.xlabel("Polynomial degree")
plt.ylabel("RMSE")
plt.legend()
plt.show()

#Omitting the last two degrees (3 and 4)

x1 = np.linspace(0,4,len(RMSE_test_error_veca)-1)
plt.plot(x1,RMSE_test_error_veca[:-1],'b',label = "Test")
plt.plot(x1,RMSE_training_error_veca[:-1],'r',label = 'Training')
plt.xlabel("Polynomial degree")
plt.ylabel("RMSE")
plt.legend()
plt.show()


###########################################################################
##########################################################################

#Part b)

RMSE_training_error_vecb = []
RMSE_test_error_vecb = []

#case1: 
X = features[:-Nsplit]
Y = labels[:-Nsplit]
M = len(X)
xtrain_set1 = X[:-round(M*0.80)]
ytrain_set1 = Y[:-round(M*0.80)]

#This part normalizes the data set 

# This function computes the mean of a column

def Avgfinder(vec_feature):
	avg_feature = np.mean(vec_feature)
	return avg_feature

#This function computes the standard deviation of a column
def Stdevfinder(vec_feature):
		stdev = np.std(vec_feature)
		return stdev

# Function that gives a 1 by 13 array with the mean of each feature of training set

def feature_mean(vec_feature):
	prearr_featuremean = []
	for k in range(len(vec_feature)):
		meanfeature = Avgfinder(vec_feature[k])
		prearr_featuremean.append(meanfeature)
	featuremean = np.array(prearr_featuremean)
	return featuremean

# Function that gives 1 by 13 array with the stdev  of each feature of training set

def feature_stdev(vec_feature):
	prearr_feature_stdev = []
	for k in range(len(vec_feature)):
		stdev_feature = Stdevfinder(vec_feature[k])
		prearr_feature_stdev.append(stdev_feature)
	feature_stdev = np.array(prearr_feature_stdev)
	return feature_stdev

xtrain1_feature_mean = feature_mean(np.transpose(xtrain_set1))
xtrain1_feature_stdev = feature_stdev(np.transpose(xtrain_set1))

#print(xtrain_feature_mean)
#print(xtrain_feature_stdev)

# Function that inputs a matrix (with feature columns) and normalizes the vector 
#with respect to this mean and stdev.
def Normfeature(matrix):
	matrix = np.transpose(matrix)
	M = matrix.shape[0] 
	N = matrix.shape[1]
	normalized_matrix_list = []
	for i in range(0,M):
		if xtrain1_feature_stdev[i] != 0:
			fi_mean_vector = np.ones(N)*xtrain1_feature_mean[i]
			row_i = (matrix[i] - fi_mean_vector)/xtrain1_feature_stdev[i]
			normalized_matrix_list.append(row_i)
		else:
			row_i = (matrix[i] - fi_mean_vector)
			normalized_matrix_list.append(row_i)
		normalized_matrix = np.transpose(np.array(normalized_matrix_list))
	return normalized_matrix

# Normalized data

# Normalized training set
xtrain1_set_normalized = Normfeature(xtrain_set1)

# Normalized test set
xtest1_set_normalized = Normfeature(X_test)

#Before going to part b,c,d,e we append a bias term to the training set.

def bias_add(vec_feature):
	if len(vec_feature) == 1:
		np_bias_vec_feature = np.insert(vec_feature, 0, 1)
	else:
		bias_vec_feature = []
		for k in range(len(vec_feature)):
			vector_k = np.insert(vec_feature[k], 0, 1)
			bias_vec_feature.append(vector_k)
		np_bias_vec_feature = np.array(bias_vec_feature)
	return np_bias_vec_feature

bias_xtrain1_set_normalized = bias_add(xtrain1_set_normalized)
bias_xtest1_set_normalized = bias_add(xtest1_set_normalized)
#print(bias_xtrain1_set_normalized.shape)

# Weight vector for features of order 1
w1_featuredeg20 = w_parameter_closed(bias_xtrain1_set_normalized,ytrain_set1)

#print(feature)

#RMSE error for feature of degree 1 using training set
rmse_featuredeg1_train_20 = rmse(bias_xtrain1_set_normalized,
	ytrain_set1,w1_featuredeg20)

RMSE_training_error_vecb.append(rmse_featuredeg1_train_20)

print("This is the RMSE for training with 20% data " + str(rmse_featuredeg1_train_20))

# RMSE error for feature of degree 1 using test set

rmse_featuredeg1_test_20 = rmse(bias_xtest1_set_normalized,y_test,w1_featuredeg20)
RMSE_test_error_vecb.append(rmse_featuredeg1_test_20)

print("This is the RMSE for test with 20% data " + str(rmse_featuredeg1_test_20))


#Case 2:

xtrain_set2 = X[:-round(M*0.60)]
ytrain_set2 = Y[:-round(M*0.60)]


def Avgfinder(vec_feature):
	avg_feature = np.mean(vec_feature)
	return avg_feature

#This function computes the standard deviation of a column
def Stdevfinder(vec_feature):
		stdev = np.std(vec_feature)
		return stdev

# Function that gives a 1 by 13 array with the mean of each feature of training set

def feature_mean(vec_feature):
	prearr_featuremean = []
	for k in range(len(vec_feature)):
		meanfeature = Avgfinder(vec_feature[k])
		prearr_featuremean.append(meanfeature)
	featuremean = np.array(prearr_featuremean)
	return featuremean

# Function that gives 1 by 13 array with the stdev  of each feature of training set

def feature_stdev(vec_feature):
	prearr_feature_stdev = []
	for k in range(len(vec_feature)):
		stdev_feature = Stdevfinder(vec_feature[k])
		prearr_feature_stdev.append(stdev_feature)
	feature_stdev = np.array(prearr_feature_stdev)
	return feature_stdev

xtrain2_feature_mean = feature_mean(np.transpose(xtrain_set2))
xtrain2_feature_stdev = feature_stdev(np.transpose(xtrain_set2))

#print(xtrain_feature_mean)
#print(xtrain_feature_stdev)

# Function that inputs a matrix (with feature columns) and normalizes the vector 
#with respect to this mean and stdev.
def Normfeature(matrix):
	matrix = np.transpose(matrix)
	M = matrix.shape[0] 
	N = matrix.shape[1]
	normalized_matrix_list = []
	for i in range(0,M):
		if xtrain2_feature_stdev[i] != 0:
			fi_mean_vector = np.ones(N)*xtrain2_feature_mean[i]
			row_i = (matrix[i] - fi_mean_vector)/xtrain2_feature_stdev[i]
			normalized_matrix_list.append(row_i)
		else:
			row_i = (matrix[i] - fi_mean_vector)
			normalized_matrix_list.append(row_i)
		normalized_matrix = np.transpose(np.array(normalized_matrix_list))
	return normalized_matrix

# Normalized data

# Normalized training set
xtrain2_set_normalized = Normfeature(xtrain_set2)

# Normalized test set
xtest2_set_normalized = Normfeature(X_test)

#Before going to part b,c,d,e we append a bias term to the training set.

def bias_add(vec_feature):
	if len(vec_feature) == 1:
		np_bias_vec_feature = np.insert(vec_feature, 0, 1)
	else:
		bias_vec_feature = []
		for k in range(len(vec_feature)):
			vector_k = np.insert(vec_feature[k], 0, 1)
			bias_vec_feature.append(vector_k)
		np_bias_vec_feature = np.array(bias_vec_feature)
	return np_bias_vec_feature

bias_xtrain2_set_normalized = bias_add(xtrain2_set_normalized)
bias_xtest2_set_normalized = bias_add(xtest2_set_normalized)

# Weight vector for features of order 2
w2_featuredeg40 = w_parameter_closed(bias_xtrain2_set_normalized,ytrain_set2)

#RMSE error for feature of degree 1 using training set
rmse_featuredeg2_train_40 = rmse(bias_xtrain2_set_normalized,
	ytrain_set2,w2_featuredeg40)

RMSE_training_error_vecb.append(rmse_featuredeg2_train_40)

print("This is the RMSE for training with 40% data " + str(rmse_featuredeg2_train_40))

# RMSE error for feature of degree 1 using test set

rmse_featuredeg2_test_40 = rmse(bias_xtest2_set_normalized,y_test,w2_featuredeg40)

RMSE_test_error_vecb.append(rmse_featuredeg2_test_40)

print("This is the RMSE for test with 40% data  " + str(rmse_featuredeg2_test_40))

# Case 3: 

xtrain_set3 = X[:-round(M*0.40)]
ytrain_set3 = Y[:-round(M*0.40)]


def Avgfinder(vec_feature):
	avg_feature = np.mean(vec_feature)
	return avg_feature

#This function computes the standard deviation of a column
def Stdevfinder(vec_feature):
		stdev = np.std(vec_feature)
		return stdev

# Function that gives a 1 by 13 array with the mean of each feature of training set

def feature_mean(vec_feature):
	prearr_featuremean = []
	for k in range(len(vec_feature)):
		meanfeature = Avgfinder(vec_feature[k])
		prearr_featuremean.append(meanfeature)
	featuremean = np.array(prearr_featuremean)
	return featuremean

# Function that gives 1 by 13 array with the stdev  of each feature of training set

def feature_stdev(vec_feature):
	prearr_feature_stdev = []
	for k in range(len(vec_feature)):
		stdev_feature = Stdevfinder(vec_feature[k])
		prearr_feature_stdev.append(stdev_feature)
	feature_stdev = np.array(prearr_feature_stdev)
	return feature_stdev

xtrain3_feature_mean = feature_mean(np.transpose(xtrain_set3))
xtrain3_feature_stdev = feature_stdev(np.transpose(xtrain_set3))

#print(xtrain_feature_mean)
#print(xtrain_feature_stdev)

# Function that inputs a matrix (with feature columns) and normalizes the vector 
#with respect to this mean and stdev.
def Normfeature(matrix):
	matrix = np.transpose(matrix)
	M = matrix.shape[0] 
	N = matrix.shape[1]
	normalized_matrix_list = []
	for i in range(0,M):
		if xtrain3_feature_stdev[i] != 0:
			fi_mean_vector = np.ones(N)*xtrain3_feature_mean[i]
			row_i = (matrix[i] - fi_mean_vector)/xtrain3_feature_stdev[i]
			normalized_matrix_list.append(row_i)
		else:
			row_i = (matrix[i] - fi_mean_vector)
			normalized_matrix_list.append(row_i)
		normalized_matrix = np.transpose(np.array(normalized_matrix_list))
	return normalized_matrix

# Normalized data

# Normalized training set
xtrain3_set_normalized = Normfeature(xtrain_set3)

# Normalized test set
xtest3_set_normalized = Normfeature(X_test)

#Before going to part b,c,d,e we append a bias term to the training set.

def bias_add(vec_feature):
	if len(vec_feature) == 1:
		np_bias_vec_feature = np.insert(vec_feature, 0, 1)
	else:
		bias_vec_feature = []
		for k in range(len(vec_feature)):
			vector_k = np.insert(vec_feature[k], 0, 1)
			bias_vec_feature.append(vector_k)
		np_bias_vec_feature = np.array(bias_vec_feature)
	return np_bias_vec_feature

bias_xtrain3_set_normalized = bias_add(xtrain3_set_normalized)
bias_xtest3_set_normalized = bias_add(xtest3_set_normalized)

# Weight vector for features of order 2
w3_featuredeg60 = w_parameter_closed(bias_xtrain3_set_normalized,ytrain_set3)

#RMSE error for feature of degree 1 using training set
rmse_featuredeg3_train_60 = rmse(bias_xtrain3_set_normalized,
	ytrain_set3,w3_featuredeg60)

RMSE_training_error_vecb.append(rmse_featuredeg3_train_60)

print("This is the RMSE for training with 60% data " + str(rmse_featuredeg3_train_60))

# RMSE error for feature of degree 1 using test set

rmse_featuredeg3_test_60 = rmse(bias_xtest3_set_normalized,y_test,w3_featuredeg60)

RMSE_test_error_vecb.append(rmse_featuredeg3_test_60)

print("This is the RMSE for test with 60% data " + str(rmse_featuredeg3_test_60))

# Case 4:

xtrain_set4 = X[:-round(M*0.20)]
ytrain_set4 = Y[:-round(M*0.20)]


def Avgfinder(vec_feature):
	avg_feature = np.mean(vec_feature)
	return avg_feature

#This function computes the standard deviation of a column
def Stdevfinder(vec_feature):
		stdev = np.std(vec_feature)
		return stdev

# Function that gives a 1 by 13 array with the mean of each feature of training set

def feature_mean(vec_feature):
	prearr_featuremean = []
	for k in range(len(vec_feature)):
		meanfeature = Avgfinder(vec_feature[k])
		prearr_featuremean.append(meanfeature)
	featuremean = np.array(prearr_featuremean)
	return featuremean

# Function that gives 1 by 13 array with the stdev  of each feature of training set

def feature_stdev(vec_feature):
	prearr_feature_stdev = []
	for k in range(len(vec_feature)):
		stdev_feature = Stdevfinder(vec_feature[k])
		prearr_feature_stdev.append(stdev_feature)
	feature_stdev = np.array(prearr_feature_stdev)
	return feature_stdev

xtrain4_feature_mean = feature_mean(np.transpose(xtrain_set4))
xtrain4_feature_stdev = feature_stdev(np.transpose(xtrain_set4))

#print(xtrain_feature_mean)
#print(xtrain_feature_stdev)

# Function that inputs a matrix (with feature columns) and normalizes the vector 
#with respect to this mean and stdev.
def Normfeature(matrix):
	matrix = np.transpose(matrix)
	M = matrix.shape[0] 
	N = matrix.shape[1]
	normalized_matrix_list = []
	for i in range(0,M):
		if xtrain4_feature_stdev[i] != 0:
			fi_mean_vector = np.ones(N)*xtrain4_feature_mean[i]
			row_i = (matrix[i] - fi_mean_vector)/xtrain4_feature_stdev[i]
			normalized_matrix_list.append(row_i)
		else:
			row_i = (matrix[i] - fi_mean_vector)
			normalized_matrix_list.append(row_i)
		normalized_matrix = np.transpose(np.array(normalized_matrix_list))
	return normalized_matrix

# Normalized data

# Normalized training set
xtrain4_set_normalized = Normfeature(xtrain_set4)

# Normalized test set
xtest4_set_normalized = Normfeature(X_test)

#Before going to part b,c,d,e we append a bias term to the training set.

def bias_add(vec_feature):
	if len(vec_feature) == 1:
		np_bias_vec_feature = np.insert(vec_feature, 0, 1)
	else:
		bias_vec_feature = []
		for k in range(len(vec_feature)):
			vector_k = np.insert(vec_feature[k], 0, 1)
			bias_vec_feature.append(vector_k)
		np_bias_vec_feature = np.array(bias_vec_feature)
	return np_bias_vec_feature

bias_xtrain4_set_normalized = bias_add(xtrain4_set_normalized)
bias_xtest4_set_normalized = bias_add(xtest4_set_normalized)

# Weight vector for features of order 4
w4_featuredeg80 = w_parameter_closed(bias_xtrain4_set_normalized,ytrain_set4)

#RMSE error for feature of degree 1 using training set
rmse_featuredeg4_train_80 = rmse(bias_xtrain4_set_normalized,
	ytrain_set4,w4_featuredeg80)

RMSE_training_error_vecb.append(rmse_featuredeg4_train_80)

print("This is the RMSE for training with 80% data " + str(rmse_featuredeg4_train_80))

# RMSE error for feature of degree 1 using test set

rmse_featuredeg4_test_80 = rmse(bias_xtest4_set_normalized,y_test,w4_featuredeg80)

RMSE_test_error_vecb.append(rmse_featuredeg4_test_80)

print("This is the RMSE for test with 80% data " + str(rmse_featuredeg4_test_80))

# Case 5:

xtrain_set5 = X
ytrain_set5 = Y

def Avgfinder(vec_feature):
	avg_feature = np.mean(vec_feature)
	return avg_feature

#This function computes the standard deviation of a column
def Stdevfinder(vec_feature):
		stdev = np.std(vec_feature)
		return stdev

# Function that gives a 1 by 13 array with the mean of each feature of training set

def feature_mean(vec_feature):
	prearr_featuremean = []
	for k in range(len(vec_feature)):
		meanfeature = Avgfinder(vec_feature[k])
		prearr_featuremean.append(meanfeature)
	featuremean = np.array(prearr_featuremean)
	return featuremean

# Function that gives 1 by 13 array with the stdev  of each feature of training set

def feature_stdev(vec_feature):
	prearr_feature_stdev = []
	for k in range(len(vec_feature)):
		stdev_feature = Stdevfinder(vec_feature[k])
		prearr_feature_stdev.append(stdev_feature)
	feature_stdev = np.array(prearr_feature_stdev)
	return feature_stdev

xtrain5_feature_mean = feature_mean(np.transpose(xtrain_set5))
xtrain5_feature_stdev = feature_stdev(np.transpose(xtrain_set5))

#print(xtrain_feature_mean)
#print(xtrain_feature_stdev)

# Function that inputs a matrix (with feature columns) and normalizes the vector 
#with respect to this mean and stdev.
def Normfeature(matrix):
	matrix = np.transpose(matrix)
	M = matrix.shape[0] 
	N = matrix.shape[1]
	normalized_matrix_list = []
	for i in range(0,M):
		if xtrain5_feature_stdev[i] != 0:
			fi_mean_vector = np.ones(N)*xtrain5_feature_mean[i]
			row_i = (matrix[i] - fi_mean_vector)/xtrain5_feature_stdev[i]
			normalized_matrix_list.append(row_i)
		else:
			row_i = (matrix[i] - fi_mean_vector)
			normalized_matrix_list.append(row_i)
		normalized_matrix = np.transpose(np.array(normalized_matrix_list))
	return normalized_matrix

# Normalized data

# Normalized training set
xtrain5_set_normalized = Normfeature(xtrain_set5)

# Normalized test set
xtest5_set_normalized = Normfeature(X_test)

#Before going to part b,c,d,e we append a bias term to the training set.

def bias_add(vec_feature):
	if len(vec_feature) == 1:
		np_bias_vec_feature = np.insert(vec_feature, 0, 1)
	else:
		bias_vec_feature = []
		for k in range(len(vec_feature)):
			vector_k = np.insert(vec_feature[k], 0, 1)
			bias_vec_feature.append(vector_k)
		np_bias_vec_feature = np.array(bias_vec_feature)
	return np_bias_vec_feature

bias_xtrain5_set_normalized = bias_add(xtrain5_set_normalized)
bias_xtest5_set_normalized = bias_add(xtest5_set_normalized)

# Weight vector for features of order 5
w5_featuredeg100 = w_parameter_closed(bias_xtrain5_set_normalized,ytrain_set5)

#RMSE error for feature of degree 1 using training set
rmse_featuredeg5_train_100 = rmse(bias_xtrain5_set_normalized,
	ytrain_set5,w5_featuredeg100)

RMSE_training_error_vecb.append(rmse_featuredeg5_train_100)

print("This is the RMSE for training with 100% data " + str(rmse_featuredeg5_train_100))

# RMSE error for feature of degree 1 using test set

rmse_featuredeg5_test_100 = rmse(bias_xtest5_set_normalized,y_test,w5_featuredeg100)

RMSE_test_error_vecb.append(rmse_featuredeg5_train_100)

print("This is the RMSE for test with 100% data " +str(rmse_featuredeg5_test_100))

#Plotting 
x1 = np.linspace(0,100,len(RMSE_training_error_vecb))
plt.plot(x1,RMSE_training_error_vecb,'b',label = "Training")
plt.plot(x1,RMSE_test_error_vecb,'r', label = "Test")
plt.xlabel("% training set")
plt.ylabel("MSE")
plt.legend()
plt.show()