import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import math
import random
from random import seed
from random import randrange, shuffle

random.seed(9001)

# Load dataset
dataset = datasets.load_boston()

features = dataset.data
labels = dataset.target

Nsplit = 50
# Training set
X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
# Test set
X_test, y_test = features[-Nsplit:], labels[-Nsplit:]

#Normalizing the features
N = len(features[1])


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
	
# Function that inputs a matrix (with feature columns) and normalizes the vector 
#with respect to this mean and stdev.
def Normfeature(matrix):
	matrix = np.transpose(matrix)
	M = matrix.shape[0] 
	N = matrix.shape[1]
	normalized_matrix_list = []
	for i in range(0,M):
		if xtrain_feature_stdev[i] != 0:
			fi_mean_vector = np.ones(N)*xtrain_feature_mean[i]
			row_i = (matrix[i] - fi_mean_vector)/xtrain_feature_stdev[i]
			normalized_matrix_list.append(row_i)
		else:
			row_i = (matrix[i] - fi_mean_vector)
			normalized_matrix_list.append(row_i)
		normalized_matrix = np.transpose(np.array(normalized_matrix_list))
	return normalized_matrix

# Normalized data

# Normalized training set
xtrain_set_normalized = Normfeature(X_train)

# Normalized test set
xtest_set_normalized = Normfeature(X_test)

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
# as input a matrix of features X, a real output vector and a weight vector w

def mse(X,Y,w):
	M = X.shape[0]
	total_sum = 0
	for d in range(M):
		yhat = np.dot(np.transpose(w),X[d])
		squared_difference = np.square(yhat - Y[d])
		total_sum = total_sum + squared_difference
	mse_result = total_sum/M
	return mse_result

x1_train = np.linspace(-0.5,500, 500)

# Problem part b) Implementing Stochastic Gradient Descent

# Recall from the setup of the problem, we are minimizing a RMSE function

w0_stoch = np.random.uniform(-0.1,0.1,N + 1);
learning_rate_grad_desc = 5e-4;	
epochs_grad_desc = 500;
mse_sgd_epoch = []

def Stochastic_grad_desc_linear_reg(matrix_data,w0,learning_rate,epochs, original_output):
	w_stoch_grad_old = w0
	M = matrix_data.shape[0]
	N = matrix_data.shape[1]
	indices = [number for number in range(M)]
	e = 0
	for e in range(epochs_grad_desc):
		random.shuffle(indices)
		for i in indices:
			diff_grad_desc = np.dot(np.transpose(w_stoch_grad_old),matrix_data[i])-original_output[i]
			stoch_grad_desc = np.dot(diff_grad_desc,matrix_data[i])
			w_stoch_grad_old = w_stoch_grad_old - learning_rate*stoch_grad_desc
		mse_sgd_epoch.append(mse(bias_xtrain_set_normalized,y_train,w_stoch_grad_old))
		#plt.plot(np.array(x1), np.array(mse_epoch), label = "Stochastic_gradient_descent")
	e = e+1
	return w_stoch_grad_old

weight_stochastic_grad_desc = Stochastic_grad_desc_linear_reg(bias_xtrain_set_normalized,
	w0_stoch,learning_rate_grad_desc,epochs_grad_desc,y_train)

# Plotting the training error for every epoch

#plt.plot(x1_train,mse_sgd_epoch, label= "MSE training error - SGD")
#plt.xlabel("Epoch #")
#plt.ylabel("MSE")
#plt.legend()
#plt.show()

print("The weight for the stochastic grad desc is " 
	+ str(weight_stochastic_grad_desc))

print("The bias term is " + str(weight_stochastic_grad_desc[0]))

# MSE for training test
MSE_trainig = mse(bias_xtrain_set_normalized, y_train, weight_stochastic_grad_desc)
print("The MSE for the training data is " + str(MSE_trainig))

# MSE for test set
MSE_test = mse(bias_xtest_set_normalized, y_test, weight_stochastic_grad_desc)
print("The MSE for the test data is " + str(MSE_test))

# Problem part c) Implementing batch gradient descent

#From the setup of the problem, we have the following parameters: 
w0_batch = np.random.uniform(-0.1,0.1,N+1)
learning_rate_batch = 5e-4
epochs_batch = 500
mse_batch_epoch = []

def batch_grad_desc(w0, matrix_data, original_output, learning_rate, epochs):
	w_batch_old = w0
	M = matrix_data.shape[0]
	N = matrix_data.shape[1]
	e = 0
	for e in range(epochs_grad_desc):
		batch_grad_desc = 0
		for i in range(M):
			diff_grad_desc = np.dot(np.transpose(w_batch_old),matrix_data[i])-original_output[i]
			pre_grad_desc = np.dot(diff_grad_desc,matrix_data[i])
			batch_grad_desc = batch_grad_desc + pre_grad_desc
		w_batch_old = w_batch_old - learning_rate*batch_grad_desc
		mse_batch_epoch.append(mse(bias_xtrain_set_normalized,y_train,w_batch_old))
		e = e + 1
	return w_batch_old

# Batch gradient descent weight vector
weight_final_batch = batch_grad_desc(w0_batch,bias_xtrain_set_normalized, 
	y_train,learning_rate_batch,epochs_batch)

#print(np.array(mse_batch_epoch).shape)

#plt.plot(x1_train,mse_batch_epoch,label = "MSE training error - Batch Gradient Descent")
#plt.xlabel("Epoch #")
#plt.ylabel("MSE")
#plt.legend()
#plt.show()

print("The weight for the batch grad desc is " + str(weight_final_batch))

# MSE for training test
MSE_trainig = mse(bias_xtrain_set_normalized, y_train, weight_final_batch)
print("The MSE training error for batch grad desc is " + str(MSE_trainig))

# MSE for test set
MSE_test = mse(bias_xtest_set_normalized, y_test, weight_final_batch)
print("The MSE test error for batch grad desc is " + str(MSE_test))

# Problem part d)

def w_parameter_closed(feature_matrix,y_train):
	XTX = np.dot(np.transpose(feature_matrix),feature_matrix)
	XTX_inv = np.linalg.inv(XTX)
	pseudo_inverse = np.dot(XTX_inv,np.transpose(feature_matrix))
	weight_vector = np.dot(pseudo_inverse,y_train)
	return weight_vector

closed_approx_weight = w_parameter_closed(bias_xtrain_set_normalized,y_train)

print("The weight for the closed form is " + str(closed_approx_weight))

# MSE for training test
MSE_trainig = mse(bias_xtrain_set_normalized, y_train, closed_approx_weight)
print("The MSE training error for closed form is " + str(MSE_trainig))

# MSE for test set
MSE_test = mse(bias_xtest_set_normalized, y_test, closed_approx_weight)
print("The MSE test error for closed form is " + str(MSE_test))

# Problem part e)

# Original features
features_orig = dataset.data
labels_orig = dataset.target
Ndata = len(features_orig)

train_errs = []
test_errs = []

for k in range(100):
	# Shuffle data
  	rand_perm = np.random.permutation(Ndata)
  	features = [features_orig[ind] for ind in rand_perm]
  	labels = [labels_orig[ind] for ind in rand_perm]

  	# Train/test split
  	Nsplit = 50
  	X_e_train, y_e_train = features[:-Nsplit], labels[:-Nsplit]
  	X_e_test, y_e_test = features[-Nsplit:], labels[-Nsplit:]

  	# Preprocess your data - Normalization, adding a constant feature
  	#params = preproc_params(X_train)
  	#X_train = preprocess(X_train, params)
  	#X_test = preprocess(X_test, params)
  	xe_train_feature_mean = feature_mean(np.transpose(X_e_train))
  	xe_train_feature_stdev = feature_stdev(np.transpose(X_e_train))

  	def Norm_e_feature(matrix):
  		matrix = np.transpose(matrix)
  		M = matrix.shape[0] 
  		N = matrix.shape[1]
  		normalized_e_matrix_list = []
  		for i in range(0,M):
  			if xe_train_feature_stdev[i] != 0:
  				fi_mean_vector = np.ones(N)*xe_train_feature_mean[i]
  				row_i = (matrix[i] - fi_mean_vector)/xe_train_feature_stdev[i]
  				normalized_e_matrix_list.append(row_i)
  			else:
  				row_i = (matrix[i] - fi_mean_vector)
  				normalized_e_matrix_list.append(row_i)
  			normalized_e_matrix = np.transpose(np.array(normalized_e_matrix_list))
  		return normalized_e_matrix

  	# Normalized training set
  	x_e_train_set_normalized = Norm_e_feature(X_e_train)
 
  	# Normalized test set
  	x_e_test_set_normalized = Norm_e_feature(X_e_test)

  	# Adding bias term to the feature matrix
  	X_e_train = bias_add(x_e_train_set_normalized)
  	X_e_test = bias_add(x_e_test_set_normalized)

  	# Solve for optimal w
  	# Use your solver function

  	w = w_parameter_closed(X_e_train, y_e_train)

  	# Collect train and test errors
  	# Use your implementation of the mse function
  	train_errs.append(mse(X_e_train, y_e_train, w))
  	test_errs.append(mse(X_e_test, y_e_test, w))

print('Mean training error: ', np.mean(train_errs))
print('Mean test error: ', np.mean(test_errs))

