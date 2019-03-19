import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import math

# Load dataset
dataset = datasets.load_boston()

features = dataset.data
labels = dataset.target

Nsplit = 50

# Training set
X = features[:-Nsplit]
Y = labels[:-Nsplit]

# Test set
X_test, y_test = features[-Nsplit:], labels[-Nsplit:]

M = len(X)

#Dividing the training set into a new training set and a validation set. 

xtrain_set, ytrain_set = X[:-round(M*0.1)], Y[:-round(M*0.1)] 
xvalidation_set, yvalidation_set = X[-round(M*0.1):], Y[-round(M*0.1):]

#print(len(xtrain_set))

N = len(xtrain_set)


#####################################################################
#####################################################################

#Preprocessing to normalize the training, validation and test.

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

xtrain_feature_mean = feature_mean(np.transpose(xtrain_set))
xtrain_feature_stdev = feature_stdev(np.transpose(xtrain_set))

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
xtrain_set_normalized = Normfeature(xtrain_set)

# Normalized validation set

xvalidation_set_normalized = Normfeature(xvalidation_set)

# Normalized test set

xtest_set_normalized = Normfeature(X_test)
																													
##############################################################################
##############################################################################

# Reuglarization parameters
lambda_parameter1 = 0
lambda_parameter2 = 0.1
lambda_parameter3 = 0.2
lambda_parameter4 = 0.3
lambda_parameter5 = 0.4
lambda_parameter6 = 0.5

# Function that adds a bias term to the feature vector all data entries

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

# Function that given the regularization parameter, a vector of features and a 
# vector of labels, returns a 456 x 13 matrix of weights w, for all the houses.
def w_parameter(lambda_param, vec_feature, vec_label):
	bias_vector = bias_add(vec_feature)
	XTX = np.dot(np.transpose(bias_vector),bias_vector)
	XTXN = XTX/N
	Id_matrix = np.identity(len(bias_vector[0]))
	XTXN_lambda = XTXN + lambda_param*Id_matrix
	XTXN_lambda_inv = np.linalg.inv(XTXN_lambda)
	XYN = np.dot(np.transpose(bias_vector),vec_label)/N
	weight = np.dot(XTXN_lambda_inv,XYN)
	arr_weight = np.array(weight)
	return arr_weight

# RMSE function: This function takes as input a weight vector, the true output 
# values and the X matrix of features and samples and returns the RMSE error:

def rmse(X,Y,w):
	M = X.shape[0]
	total_sum = 0
	for d in range(M):
		yhat = np.dot(np.transpose(w),X[d])
		squared_difference = np.square(yhat - Y[d])
		total_sum = total_sum + squared_difference
	rmse_result = np.sqrt(total_sum/M)
	return rmse_result

# Empty lists preparing for plots
RMSE_training_vec = []
RMSE_test_vec = []
RMSE_validation_vec = []

# This is the analysis for Lambda = 0 (lambda1). For each case, we report 
# the loss on validation set, the lowest RMSE and the test error.

# Weight vector for lambda = 0
w1 = w_parameter(lambda_parameter1, xtrain_set_normalized, ytrain_set)
print(w1)

#Appending the bias value 1 to the training set
bias_xtraining_set1 = bias_add(xtrain_set_normalized)
RMSE_training1 = rmse(bias_xtraining_set1, ytrain_set, w1)
RMSE_training_vec.append(RMSE_training1)

# Now we do a similar process for validation set using the RMSE function.
bias_xvalidation_set1 = bias_add(xvalidation_set_normalized)
RMSE_validation1 = rmse(bias_xvalidation_set1, yvalidation_set,w1)
RMSE_validation_vec.append(RMSE_validation1)

# Finally, we do a similar process for test set:
bias_xtest_set1 = bias_add(xtest_set_normalized)
RMSE_test1 = rmse(bias_xtest_set1, y_test,w1)
RMSE_test_vec.append(RMSE_test1)

print('The RMSE error of training set with lambda = 0 is ' + str(RMSE_training1))
print('The RMSE error of validation set with lambda = 0 is ' + str(RMSE_validation1))
print('The RMSE error of test set with lambda = 0 is ' + str(RMSE_test1))


# This is the analysis for Lambda = 0.1 (lambda2). For each case, we report 
# the loss on validation set, the lowest RMSE and the test error.

# Weight vector for lambda = 0.1
w2 = w_parameter(lambda_parameter2, xtrain_set_normalized, ytrain_set)

#Appending the bias value 1 to the training set
bias_xtraining_set2 = bias_add(xtrain_set_normalized)
RMSE_training2 = rmse(bias_xtraining_set2, ytrain_set, w2)
RMSE_training_vec.append(RMSE_training2)

# Now we do a similar process for validation set using the RMSE function.
bias_xvalidation_set2 = bias_add(xvalidation_set_normalized)
RMSE_validation2 = rmse(bias_xvalidation_set2, yvalidation_set,w2)
RMSE_validation_vec.append(RMSE_validation2)

# Finally, we do a similar process for test set:
bias_xtest_set2 = bias_add(xtest_set_normalized)
RMSE_test2 = rmse(bias_xtest_set2, y_test,w2)
RMSE_test_vec.append(RMSE_test2)

print('The RMSE error of training set with lambda = 0.1 is ' + str(RMSE_training2))
print('The RMSE error of validation set with lambda = 0.1 is ' + str(RMSE_validation2))
print('The RMSE error of test set with lambda = 0.1 is ' + str(RMSE_test2))

# This is the analysis for Lambda = 0.2 (lambda3). For each case, we report 
# the loss on validation set, the lowest RMSE and the test error.

# Weight vector for lambda = 0.2
w3 = w_parameter(lambda_parameter3, xtrain_set_normalized, ytrain_set)

#Appending the bias value 1 to the training set
bias_xtraining_set3 = bias_add(xtrain_set_normalized)
RMSE_training3 = rmse(bias_xtraining_set3, ytrain_set, w3)
RMSE_training_vec.append(RMSE_training3)

# Now we do a similar process for validation set using the RMSE function.
bias_xvalidation_set3 = bias_add(xvalidation_set_normalized)
RMSE_validation3 = rmse(bias_xvalidation_set3, yvalidation_set,w3)
RMSE_validation_vec.append(RMSE_validation3)

# Finally, we do a similar process for test set:
bias_xtest_set3 = bias_add(xtest_set_normalized)
RMSE_test3 = rmse(bias_xtest_set3, y_test,w3)
RMSE_test_vec.append(RMSE_test3)

print('The RMSE error of training set with lambda = 0.2 is ' + str(RMSE_training3))
print('The RMSE error of validation set with lambda = 0.2 is ' + str(RMSE_validation3))
print('The RMSE error of test set with lambda = 0.2 is ' + str(RMSE_test3))


# This is the analysis for Lambda = 0.3 (lambda4). For each case, we report 
# the loss on validation set, the lowest RMSE and the test error.

# Weight vector for lambda = 0.3
w4 = w_parameter(lambda_parameter4, xtrain_set_normalized, ytrain_set)

#Appending the bias value 1 to the training set
bias_xtraining_set4 = bias_add(xtrain_set_normalized)
RMSE_training4 = rmse(bias_xtraining_set4, ytrain_set, w4)
RMSE_training_vec.append(RMSE_training4)

# Now we do a similar process for validation set using the RMSE function.
bias_xvalidation_set4 = bias_add(xvalidation_set_normalized)
RMSE_validation4 = rmse(bias_xvalidation_set4, yvalidation_set,w4)
RMSE_validation_vec.append(RMSE_validation4)

# Finally, we do a similar process for test set:
bias_xtest_set4 = bias_add(xtest_set_normalized)
RMSE_test4 = rmse(bias_xtest_set4, y_test,w4)
RMSE_test_vec.append(RMSE_test4)

print('The RMSE error of training set with lambda = 0.3 is ' + str(RMSE_training4))
print('The RMSE error of validation set with lambda = 0.3 is ' + str(RMSE_validation4))
print('The RMSE error of test set with lambda = 0.3 is ' + str(RMSE_test4))

# This is the analysis for Lambda = 0.4 (lambda5). For each case, we report 
# the loss on validation set, the lowest RMSE and the test error.

# Weight vector for lambda = 0.4
w5 = w_parameter(lambda_parameter5, xtrain_set_normalized, ytrain_set)

#Appending the bias value 1 to the training set
bias_xtraining_set5 = bias_add(xtrain_set_normalized)
RMSE_training5 = rmse(bias_xtraining_set5, ytrain_set, w5)
RMSE_training_vec.append(RMSE_training5)

# Now we do a similar process for validation set using the RMSE function.
bias_xvalidation_set5 = bias_add(xvalidation_set_normalized)
RMSE_validation5 = rmse(bias_xvalidation_set5, yvalidation_set,w5)
RMSE_validation_vec.append(RMSE_validation5)

# Finally, we do a similar process for test set:
bias_xtest_set5 = bias_add(xtest_set_normalized)
RMSE_test5 = rmse(bias_xtest_set5, y_test,w5)
RMSE_test_vec.append(RMSE_test5)

print('The RMSE error of training set with lambda = 0.4 is ' + str(RMSE_training5))
print('The RMSE error of validation set with lambda = 0.4 is ' + str(RMSE_validation5))
print('The RMSE error of test set with lambda = 0.4 is ' + str(RMSE_test5))

# This is the analysis for Lambda = 0.5 (lambda6). For each case, we report 
# the loss on validation set, the lowest RMSE and the test error.

# Weight vector for lambda = 0.5
w6 = w_parameter(lambda_parameter6, xtrain_set_normalized, ytrain_set)

#Appending the bias value 1 to the training set
bias_xtraining_set6 = bias_add(xtrain_set_normalized)
RMSE_training6 = rmse(bias_xtraining_set6, ytrain_set, w6)
RMSE_training_vec.append(RMSE_training6)

# Now we do a similar process for validation set using the RMSE function.
bias_xvalidation_set6 = bias_add(xvalidation_set_normalized)
RMSE_validation6 = rmse(bias_xvalidation_set6, yvalidation_set,w6)
RMSE_validation_vec.append(RMSE_validation6)

# Finally, we do a similar process for test set:
bias_xtest_set6 = bias_add(xtest_set_normalized)
RMSE_test6 = rmse(bias_xtest_set6, y_test,w6)
RMSE_test_vec.append(RMSE_test6)

print('The RMSE error of training set with lambda = 0.5 is ' + str(RMSE_training6))
print('The RMSE error of validation set with lambda = 0.5 is ' + str(RMSE_validation6))
print('The RMSE error of test set with lambda = 0.5 is ' + str(RMSE_test6))

print('The lowest RMSE error identified in the validation set is ' + str(RMSE_validation4) + 'happening on lambda 0.3')

print('Test RMSE at best lambda ' + str(0.3) + " is " +str(RMSE_test4))

##Plotting the graph of the RMSE

x1 = np.linspace(0,0.5,6)
plt.plot(x1,RMSE_training_vec,'r--', label = "Training")
plt.plot(x1,RMSE_validation_vec,'b', label = "Validation")
plt.plot(x1,RMSE_test_vec,'g', label = "Test")
plt.xlabel("Lambda")
plt.ylabel("RMSE")
plt.legend()
plt.show()

