import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from random import uniform

# Normalized Binary dataset
# 4 features, 100 examples, 50 labeled 0 and 50 labeled 1
X, y = load_breast_cancer().data, load_breast_cancer().target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#print(y)


# Part 1 of the problem

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


# Function that inputs a matrix (with feature columns) and normalizes the vector 
#with respect to the vetors of mean and stdev given.
def Normfeature(matrix, mean, stdev):
    matrix = np.transpose(matrix)
    M = matrix.shape[0] 
    N = matrix.shape[1]
    normalized_matrix_list = []
    for i in range(0,M):
        if stdev[i] != 0:
            fi_mean_vector = np.ones(N)*mean[i]
            row_i = (matrix[i] - fi_mean_vector)/stdev[i]
            normalized_matrix_list.append(row_i)
        else:
            row_i = (matrix[i] - fi_mean_vector)
            normalized_matrix_list.append(row_i)
        normalized_matrix = np.transpose(np.array(normalized_matrix_list))
    return normalized_matrix

# Function that appends a bias term to the training set.

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

def mse(X,Y,w):
	M = X.shape[0]
	total_sum = 0
	for d in range(M):
		yhat = np.dot(np.transpose(w),X[d])
		squared_difference = np.square(yhat - Y[d])
		total_sum = total_sum + squared_difference
	mse_result = total_sum/M
	return mse_result

#Getting the mean and standard deviation of each column of the training set

xtrain_feature_mean = feature_mean(np.transpose(X_train))
xtrain_feature_stdev = feature_stdev(np.transpose(X_train))

# Normalized training set
xtrain_set_normalized = Normfeature(X_train,
xtrain_feature_mean,xtrain_feature_stdev)
 
print("Normalized train set" + str(xtrain_set_normalized.shape))
#Normalized test set
xtest_set_normalized = Normfeature(X_test,
xtrain_feature_mean,xtrain_feature_stdev)

# Normalized data with bias term appended.  
bias_xtrain_set_normalized = bias_add(xtrain_set_normalized)
bias_xtest_set_normalized = bias_add(xtest_set_normalized)

#print(np.mean(bias_xtrain_set_normalized[:,0]))

def sigmoid(a):
	value = 1/(1+np.exp(-a))
	return value


# Takes as input the features matrix, a target vector and the vector of weights
def log_likelihood(feature_matrix, target_vector, weights):
	identity = target_vector.shape[0]
	ll = 0
	for i in range(identity):
		product = np.dot(np.transpose(weights),feature_matrix[i])
		ll = ll + target_vector[i]*np.log(sigmoid(product)) + (1-target_vector[i])*np.log(1-sigmoid(product))
	ll = ll/identity
	return -ll

# Takes as input the matrix of features, a target vector and the vector of weights:
def grad_likelihood(features_vector, target_vector, weights):
	#n = target_vector
	sigmoid_vector = []
	product = np.dot(features_vector,weights)
	diff = np.subtract(sigmoid(product),target_vector)
	grad = np.dot(np.transpose(features_vector),diff)
	return grad


w_initial = []
N = xtest_set_normalized.shape[1]
for i in range(N):
	w_initial.append(uniform(-1,1))

learning_rate = 1e-2
epochs = 1
mse_sgd_epoch = []
M = xtrain_set_normalized.shape[0]
indices = [number for number in range(M)]


def stochastic_grad_desc_logreg(w_initial, training_features_matrix,training_target,test_features_matrix,test_target,learning_rate,epochs):
	w = w_initial
	M = training_features_matrix.shape[0]
	N = training_features_matrix.shape[1]
	training_error = np.zeros(M)
	test_error = np.zeros(M)
	training_acc = np.zeros(M)
	test_acc = np.zeros(M)
	indices = [number for number in range(M)]
	for i in indices:
	
		#grad = np.dot(np.transpose(training_features_matrix[i]),np.subtract(sigmoid(np.dot(training_features_matrix[i],w)),training_target[i]))
		grad = grad_likelihood(training_features_matrix[i],training_target[i],w)
		w = np.subtract(w,learning_rate*grad)
		training_error[i] += log_likelihood(training_features_matrix,training_target,w)
		test_error[i] += log_likelihood(test_features_matrix,test_target,w)

		right = 0	
		for j in indices:
			P1 = sigmoid(np.dot(np.transpose(w),training_features_matrix[j]))
			if P1 >= 0.5:
				if y_train[j] == 1:
					right += 1
			else:
				if y_train[j] == 0:
					right += 1

		training_acc[i] = (right/M) * 100 

		right = 0	
		for j in range(test_features_matrix.shape[0]):
			P2 = sigmoid(np.dot(np.transpose(w),test_features_matrix[j]))
			if P2 >= 0.5:
				if y_test[j] == 1:
					right += 1
			else:
				if y_test[j] == 0:
					right += 1

		test_acc[i] = (right/test_features_matrix.shape[0]) * 100 

	return w,test_error,training_error,training_acc, test_acc


[weight_stochastic_grad_descent,training_error,test_error,training_acc,test_acc] = stochastic_grad_desc_logreg(w_initial,
		xtrain_set_normalized,y_train,xtest_set_normalized,y_test,learning_rate, epochs)

iterations_training = len(training_error)
print("This is weight vector WITHOUT bias term: " + str(weight_stochastic_grad_descent))
print("This is the final training cross entropy: " + str(training_error[iterations_training-1]))
print("This is the final test cross entropy: " + str(test_error[iterations_training-1]))
print("This is the final training classification accuracy percentage: " + str(training_acc[iterations_training-1]))
print("This is the final test classification accuracy percentage: " + str(test_acc[iterations_training-1]))
#weight_stochastic_grad_descent_no_bias = np.delete(weight_stochastic_grad_descent,weight_stochastic_grad_descent[0])
#print("This is weight vector WITHOUT bias term" + str(weight_stochastic_grad_descent_no_bias.shape))


plt.plot(indices,training_error,label = 'Average Training error')
plt.plot(indices,test_error, label = 'Average Test error')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('loss')
plt.suptitle('Average cross-entropy loss E(w) per SGD iteration')
plt.show()

plt.plot(indices,training_acc,label = 'Training accuracy')
plt.plot(indices,test_acc, label = 'Test accuracy')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('Percentage accuracy prediction')
plt.suptitle('Training and test accuracy per SGD iteration')
plt.show()



