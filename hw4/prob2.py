import numpy as np

from sklearn import datasets, svm
#fetch original mnist dataset
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import cross_val_score

mnist = fetch_mldata('MNIST original', data_home='./')

#data field is 70k x 784 array, each row represents pixels from 28x28=784 image
images = mnist.data
targets = mnist.target

N = len(images)
np.random.seed(1234)
inds = np.random.permutation(N)
images = np.array([images[i] for i in inds])
targets = np.array([targets[i] for i in inds])

# Normalize data
X_data = images/255.0
Y = targets

# Train/test split
X_train, y_train = X_data[:10000], Y[:10000]
X_test, y_test = X_data[-10000:], Y[-10000:]
print(X_test[1,:])


# First we will check the accuracy of the SVM rbf classifier

classifier = svm.SVC(C = 1, gamma = 1)
classifier.fit(X_train,y_train)

test_shape = np.array(X_test).shape
no_test_elements = test_shape[0]
#prediction = zeros(no_test_elements)

prediction = classifier.predict(X_test)

#for c in range(no_test_elements):
#	prediction[c] = classifier.predict(X_test(c))

print(prediction.shape)

sum_errors = 0

for s in range(no_test_elements):
	if prediction[s] != y_test[s]:
		sum_errors = sum_errors + 1

error_percentage = sum_errors/no_test_elements
accuracy = 1 - error_percentage
print('The accuracy of the SVM classifier with rbf kernel is ' + str(accuracy))


## This part will be for the cross validation.

data_fifth = N/5
X_0, Y_0 = X_data[:14000], Y[:14000]
X_1, Y_1 = X_data[14000:2 * 14000 ], Y[14000: 2 * 14000]
X_2, Y_2 = X_data[2*14000:3*14000 ], Y[2*14000:3*14000 ]
X_3, Y_3 = X_data[3*14000 :4*14000 ], Y[3*14000  : 4*14000 ]
X_4, Y_4 = X_data[4 * 14000 :5 *14000], Y[4*14000 :5 * 14000 ]

X_data_segments = [X_0,X_1,X_2,X_3,X_4]
Y_data_segments = [Y_0,Y_1,Y_2,Y_3,Y_4]

results = []
C = [1,3,5]
gamma = [0.05,0.1,0.5,1.0]

for x in C:
	for g in gamma:
		sum_errors_k = 0
		sum_accuracy_k = 0
		for k in range(5):
			temp_X_test = X_data_segments[k]
			temp_X_train = X_data_segments[(k+1)%5] + X_data_segments[(k+2)%5] + X_data_segments[(k+3)%5] + X_data_segments[(k+4)%5]
			temp_Y_test = Y_data_segments[k]
			temp_Y_train = Y_data_segments[(k+1)%5] + Y_data_segments[(k+2)%5] + Y_data_segments[(k+3)%5] + Y_data_segments[(k+4)%5]
			classifier = svm.SVC(C = x, gamma= g)
			classifier.fit(temp_X_train,temp_Y_train)
			prediction = classifier.predict(temp_X_test)
			sum_errors = 0
			for s in range(len(prediction)):
				if prediction[s] != temp_Y_test[s]:
					sum_errors = sum_errors + 1
			k_prediction_error = sum_errors/len(prediction)
			k_prediction_accuracy = 1 - k_prediction_error

			sum_errors_k = sum_errors_k + k_prediction_error
			sum_accuracy_k = sum_accuracy_k + k_prediction_accuracy

		avg_error = sum_errors_k / 5
		avg_accuracy = sum_accuracy_k / 5
		results.append([x,g,avg_error,avg_accuracy])

array_results = np.array(results)
accuracys = results[:,3]
min_accuracy = np.amin(accuracys)
index_min = np.where(accuracys == min_accuracy)[0][0]
winner_combo = array_results[index_min]

print('The optimal C is ' + str(winner_combo[0]))
print('The optimal g is ' + str(winner_combo[1]))