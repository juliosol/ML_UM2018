from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt

data, target = load_boston().data, load_boston().target

## Normalizing data
def data_normalizer(data):
	row,col = data.shape
	for j in range(col):
		mean = np.mean(data[:,j])
		variance = np.var(data[:,j])
		data[:,j] = np.divide(np.subtract(data[:,j],mean),(variance**(1/2)))
	return data

new_data = data_normalizer(data)
#print(np.var(new_data[:,]))

## This part is for computing the PCA 

# Computing mean of data vectors

def mean_data_features(data):
	row, col = data.shape
	sum_columns = 0
	for j in range(row):
		sum_columns = sum_columns + data[j,:]
	mean_columns = sum_columns/col
	return mean_columns

#print(mean_data_features(data).shape)

# Computing the covariance matrix

def covariance_matrix(data):
	row,col = data.shape
	mean = mean_data_features(data)
	sum_covariance = np.zeros((col,col))
	for j in range(col):
		inner_product = np.outer(np.subtract(data[j,:],mean),np.subtract(data[j,:],mean))
		sum_covariance = sum_covariance + inner_product
	covariance_matrix = sum_covariance/col
	return covariance_matrix

#print(covariance_matrix(data).shape)

#covariance_matrix_data = covariance_matrix(new_data)
row,col = data.shape
covariance_matrix_data = np.cov(np.transpose(new_data))

## Eigenvalues
w,v = np.linalg.eig(covariance_matrix_data)
e1 = w[0]
ev1 = v[0]
e2 = w[1]
ev2 = v[1]
#print(w)

### Directions
print('Direction 1 is: ' + str(ev1))
print('Direction 2 is: ' + str(ev2))

#plt.scatter(e1, e2,c=target / max(target))
#plt.show()

projection = np.zeros((row,2))
for j in range(row):
	projection[j][0] = np.dot(ev1,new_data[j][:])
	projection[j][1] = np.dot(ev2,new_data[j][:])

plt.scatter(projection[:,0],projection[:,1],c=target / max(target))
plt.show()