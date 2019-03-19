import numpy as np
import collections

# load the training data 
train_features = np.load("spam_train_features.npy")
train_labels = np.load("spam_train_labels.npy")


# load the test data 
test_features = np.load("spam_test_features.npy")
test_labels = np.load("spam_test_labels.npy")

len_labels = len(test_labels)


# Function that computes the posterior mean for class 0 and class 1

def variables(train_features,train_labels):
	no_examples_features = train_features.shape
	N = no_examples_features[0]
	no_features = no_examples_features[1]

	N_class0 = np.count_nonzero(train_labels == 1)
	N_class1 = N - N_class0

	N_class0_features_val0 = np.zeros([no_features])
	N_class1_features_val1 = np.zeros([no_features])
	N_class0_features_val1 = np.zeros([no_features])
	N_class1_features_val0 = np.zeros([no_features])

	for i in range(no_features):
		counter_class0_features_val0 = 0
		counter_class1_features_val1 = 0
		counter_class0_features_val1 = 0
		counter_class1_features_val0 = 0
		for j in range(N):
			if train_labels[j] == 0 and train_features[j,i] == 0:
				counter_class0_features_val0 = counter_class0_features_val0 + 1
			elif train_labels[j] == 0 and train_features[j,i] == 1:
				counter_class0_features_val1 = counter_class0_features_val1 + 1
			elif train_labels[j] == 1 and train_features[j,i] == 1:
				counter_class1_features_val1 = counter_class1_features_val1 + 1
			elif train_labels[j] == 1 and train_features[j,i] == 0:
				counter_class1_features_val0 = counter_class1_features_val0 + 1
		#N_class1_features_val0.append(counter_class1_features_val0)
		#np.append(N_class1_features_val0,counter_class1_features_val0)
		N_class1_features_val0[i] = counter_class1_features_val0
		#N_class0_features_val1.append(counter_class0_features_val1)
		#np.append(N_class0_features_val1,counter_class0_features_val1)
		N_class0_features_val1[i] = counter_class0_features_val1
		#N_class0_features_val0.append(counter_class0_features_val0)
		#np.append(N_class0_features_val0,counter_class0_features_val0)
		N_class0_features_val0[i] = counter_class0_features_val0
		#N_class1_features_val1.append(counter_class1_features_val1)
		#np.append(N_class1_features_val1,counter_class1_features_val1)
		N_class1_features_val1[i] = counter_class1_features_val1
		#print(N_class1_features_val1)
		#print(counter_class1_features_val1)

	#print(N_class1_features_val0.shape)

	pi_mean_post_c0 = (N_class0 + 1.0)/(N + 2.0)

	pi_mean_post_c1 = (N_class1 + 1.0)/(N+2.0)

	theta_class0_d0 = (N_class0_features_val0 + np.ones([no_features]))/(N_class0 + 1*no_features)

	theta_class1_d1 = (N_class1_features_val1 + np.ones([no_features]))/(N_class1 + 1*no_features)

	theta_class1_d0 = (N_class1_features_val0 + np.ones([no_features]))/(N_class1 + 1*no_features)

	theta_class0_d1 = (N_class0_features_val1 + np.ones([no_features]))/(N_class0 + 1*no_features)

	return pi_mean_post_c0,pi_mean_post_c1,theta_class0_d0,theta_class0_d1,theta_class1_d0,theta_class1_d1
	

# This part is the prediction portion 

def naive_b_prediction(train_features,train_labels,test_features_vector,test_labels):

	pi_mean_post_c0,pi_mean_post_c1,theta_class0_d0,theta_class0_d1,theta_class1_d0,theta_class1_d1 = variables(train_features,train_labels)
	
	pi_list = [pi_mean_post_c0, pi_mean_post_c1]
	theta_class0 = [theta_class0_d0,theta_class0_d1]
	theta_class1 = [theta_class1_d0,theta_class1_d1]

	#print(list(set(train_labels)))
	classes = set(train_labels)
	no_classes = len(list(classes))

	y = np.zeros([no_classes])

	#matrix_features_size = test_features.shape
	#no_samples_test = matrix_features_size[0]
	#no_features_test = len(test_features_vector)
	no_features_test = 2

	y_vector = np.array(range(no_classes))

	c_max = 0

	theta_cdm = 0
	theta = 0

	for c in range(no_classes):
		pi = pi_list[c]
		theta = 1
		theta_cdm = 1
		for d in range(no_features_test):
			for m in classes: 
				#if test_features_vector[d] == m:
				#	theta_cdm = theta_class0[m][d]
				theta_cdm = theta_class0[m][d]
			theta = theta * theta_cdm
		y_vector[c] = pi * theta
		if c >= 1 and c <= no_classes - 1:
			if y_vector[c] >= y_vector[c-1]:
				c_max = c
				print(c_max)
	return c_max


error = 0
for i in range(len_labels):
	prediction = naive_b_prediction(train_features,train_labels,test_features[i],test_labels[i])
	if prediction != test_labels[i]:
		error = error + 1

error_rate = error/len_labels

print(error_rate)



