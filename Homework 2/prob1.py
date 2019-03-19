import numpy as np
import matplotlib.pyplot as plt


# For this problem, we use data generator instead of real dataset
def data_generator(size,noise_scale=0.05):
    xs = np.random.uniform(low=0,high=3,size=size)

    # for function y = 0.5x - 0.3 + sin(x) + epsilon, where epsilon is a gaussian noise with std dev= 0.05
    ys = xs * 0.5 - 0.3 + np.sin(3*xs) + np.random.normal(loc=0,scale=noise_scale,size=size)
    return xs, ys

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


# Function that computes the mean squared error given predicted data X, 
# actual output data Y and weight vector w. 
def mse(X,Y,w):
    M = X.shape[0]
    total_sum = 0
    for d in range(M):
        yhat = np.dot(np.transpose(w),X[d])
        squared_difference = np.square(yhat - Y[d])
        total_sum = total_sum + squared_difference
    mse_result = total_sum/M
    return mse_result

# Function that computes the weight vector of the closed form
# linear regression

def w_parameter_closed(feature_matrix,y_train):
    XTX = np.dot(np.transpose(feature_matrix),feature_matrix)
    XTX_inv = np.linalg.inv(XTX)
    pseudo_inverse = np.dot(XTX_inv,np.transpose(feature_matrix))
    weight_vector = np.dot(pseudo_inverse,y_train)
    return weight_vector


###############################################################################
#######################################################################33333

# Part 2 of the problem


def GaussKer(x0,x,sigma):

    #We will start by writing a function GaussKer that defines a Gaussian Kernel.
    #This function receives as input:
    #x0: A point we are trying to estimate
    #x: A training point we have
    #sigma: Kernel parameter that stands for Standard Deviation of the Gaussian.
    
    diff = x - x0
    abs_diff = abs(diff)
    squared_diff = np.dot(np.transpose(abs_diff) , abs_diff)
    return np.exp(-squared_diff/(2*sigma^2))

def local_weights(sigma,training_data,data_predict):
    x = np.mat(training_data)
    n_rows = x.shape[0]
    weights = np.mat(np.eye(n_rows))
    for i in range(n_rows):
        weights[i,i] = GaussKer(data_predict,x[i],sigma)
    return weights

def loc_weighted_regression_prediction(sigma,training_inputs,training_outputs, data_to_predict):
    weights = local_weights(sigma,training_inputs, data_predict)

    x = np.mat(training_inputs)
    y = np.mat(training_outputs)

    inverse_product = np.dot(np.dot(np.transpose(x),weights),x)
    betas = np.dot(np.linalg.inv(inverse_product),np.dot(x,np.dot(weights,y)))
    predicted_data = np.dot(datapoint,betas)

def main():
    noise_scales = [0.05,0.2]

    # for example, choose the first kind of noise scale
    noise_scale = noise_scales[0]

    # generate the data form generator given noise scale
    X_train, y_train = data_generator((100,1),noise_scale=noise_scale)
    X_test, y_test = data_generator((30,1),noise_scale=noise_scale)

    #plt.scatter(X_train,y_train)
    #plt.show()

    # bandwidth parameters
    sigma_paras = [0.1,0.2,0.4,0.8,1.6]

    #Getting the mean and standard deviation of each column of the training set

    xtrain_feature_mean = feature_mean(np.transpose(X_train))
    xtrain_feature_stdev = feature_stdev(np.transpose(X_train))

    # Normalized training set
    xtrain_set_normalized = Normfeature(X_train,
        xtrain_feature_mean,xtrain_feature_stdev)
    # Normalized test set
    xtest_set_normalized = Normfeature(X_test,
        xtrain_feature_mean,xtrain_feature_stdev)

    # Normalized data with bias term appended.  
    bias_xtrain_set_normalized = bias_add(xtrain_set_normalized)
    bias_xtest_set_normalized = bias_add(xtest_set_normalized)

    # Part 1 of the problem, getting the weight vector in closed form linear regression

    #print(bias_xtrain_set_normalized.size)
    #print(y_train.size)
    closed_approx_weight_train = w_parameter_closed(bias_xtrain_set_normalized,y_train)
    closed_approx_weight_test = w_parameter_closed(bias_xtest_set_normalized,y_test)
    #print(closed_approx_weight_test.shape)
    #print(bias_xtrain_set_normalized.shape)
    approximations_train = np.dot(bias_xtrain_set_normalized,closed_approx_weight_train)
    approximations_test = np.dot(bias_xtest_set_normalized,closed_approx_weight_test)

    #plt.scatter(X_train,approximations_train)
    #plt.scatter(X_test,approximations_test)
    plt.scatter(X_test,y_test)
    plt.show()
    #print(closed_approx_weight)

    # Part 2 of the problem

    

main()

