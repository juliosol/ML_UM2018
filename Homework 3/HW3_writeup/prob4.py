from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import math

iris=load_iris()

# You have two features and two classifications
data_0, data_1 = iris.data[:,1:3][:50], iris.data[:,1:3][50:100]

# TODO: Compute the mean and covariance of each cluster, and use these to find a QDA Boundary

total_data = np.concatenate((data_0,data_1),axis = 0)
#print(total_data)

plt.scatter(data_0[:,0],data_0[:,1],color = 'blue')
plt.scatter(data_1[:,0],data_1[:,1],color = 'red')
#plt.show()


mean_data_0_col0 = np.array(np.mean(data_0[:,0]))
mean_data_0_col1 = np.array(np.mean(data_0[:,1]))
mean_data_1_col0 = np.array(np.mean(data_1[:,0]))
mean_data_1_col1 = np.array(np.mean(data_1[:,1]))

mean_data_0 = [mean_data_0_col0,mean_data_0_col1]
mean_data_1 = [mean_data_1_col0,mean_data_1_col1]

no_class_0 = data_0.shape[0]
no_class_1 = data_1.shape[0]
total_amount_data = no_class_0 + no_class_1

pre_covariance_0 = 0
for i in range(no_class_0):
	product = np.outer(np.subtract(data_0[i], mean_data_0),np.subtract(data_0[i], mean_data_0))
	pre_covariance_0 = pre_covariance_0 + product

pre_covariance_1 = 0
for i in range(no_class_1):
	product = np.outer(np.subtract(data_1[i], mean_data_1),np.subtract(data_1[i], mean_data_1))
	pre_covariance_1 = pre_covariance_1 + product


covariance_data_0 = np.divide(pre_covariance_0,no_class_0)
covariance_data_1 = np.divide(pre_covariance_1,no_class_1)

def qda(vector_data, covariance_data_0,covariance_data_1,mu_0,mu_1):
	inverse_cov_data_1 = np.linalg.inv(covariance_data_1)
	inverse_cov_data_0 = np.linalg.inv(covariance_data_0)
	difference_cov_10 = np.subtract(inverse_cov_data_1,inverse_cov_data_0)
	quad_term = 1/2*np.dot(np.dot(np.transpose(vector_data),difference_cov_10),vector_data)
	deg1_term = -2*np.dot(np.dot(np.transpose(mu_0), inverse_cov_data_0) + np.dot(np.transpose(mu_1),inverse_cov_data_1), vector_data) 
	non_deg_terms = np.dot(np.dot(np.transpose(mu_1),inverse_cov_data_1),mu_1) + np.dot(np.dot(np.transpose(mu_0),inverse_cov_data_0),mu_0) + 1/2*math.log(np.linalg.det(inverse_cov_data_1)/np.linalg.det(inverse_cov_data_0)) 
	qda_value = quad_term + deg1_term + non_deg_terms
	return qda_value

def post_qda(vector_data,covariance_data_0,covariance_data_1,mu_0,mu_1): 
	inverse_cov_data_1 = np.linalg.inv(covariance_data_1)
	inverse_cov_data_0 = np.linalg.inv(covariance_data_0)
	squared_term_0 = 1/2*np.dot(np.dot(np.transpose(np.subtract(vector_data,mu_0)),inverse_cov_data_0),np.subtract(vector_data,mu_0))
	single_terms_0 = 1/2*math.log(np.linalg.det(inverse_cov_data_0)) - math.log(0.5)
	prediction_0 = squared_term_0 + single_terms_0
	squared_term_1 = 1/2*np.dot(np.dot(np.transpose(np.subtract(vector_data,mu_1)),inverse_cov_data_1),np.subtract(vector_data,mu_1))
	single_terms_1 = 1/2*math.log(np.linalg.det(inverse_cov_data_1)) - math.log(0.5)
	prediction_1 = squared_term_1 + single_terms_1
	if prediction_0 < prediction_1:
		return 0
	else:
		return 1


#prediction = post_qda(data_0[3],covariance_data_0,covariance_data_1,mean_data_0,mean_data_1)

#print(prediction)	

# TODO: Compute the mean and covariance of the entire dataset, and use these to find a LDA Boundary

# Here we calculate the covariance for the entire data set.

#pre_covariance_tota = 0
covariance_total = np.cov(np.transpose(total_data),rowvar = True)
#print(np.linalg.inv(covariance_total))

def post_LDA(vector_data,covariance_matrix,mu_0,mu_1):
	inv_covariance = np.linalg.inv(covariance_total)
	var_term = np.dot(np.dot(np.transpose(np.subtract(mu_1,mu_0)),inv_covariance),vector_data)
	nonvar_term = -0.5*np.dot(np.dot(np.transpose(mu_1),inv_covariance),mu_1) + 0.5*np.dot(np.dot(np.transpose(mu_0),inv_covariance),mu_0)
	prediction = var_term + nonvar_term
	if prediction > 0:
		return 1
	else:
		return 0

prediction = post_LDA(data_0[3],covariance_total,mean_data_0,mean_data_1)

#print(prediction)

# TODO: Make two scatterplots of the data, one showing the QDA Boundary and one showing the LDA Boundary

# Scatterplot showing QDA boundary

no_points = 100
x = np.linspace(0,5,no_points)
y= np.linspace(0,6,no_points)
xx, yy = np.meshgrid(x,y)
x_len = x.shape[0]
y_len = y.shape[0]
Z = np.zeros((x_len, y_len))

for p in range(x_len):
	for q in range(y_len):
		data_point = [xx[q][p],yy[q][p]]
		Z[q][p] = post_qda(data_point,covariance_data_0,covariance_data_1,mean_data_0,mean_data_1)
		
#plt.contour(xx,yy,Z,1)
#plt.title('QDA')
#plt.xlabel('x values')
#plt.ylabel('y values')
#plt.show()

# Scatterplot showing QDA boundary

no_points = 100
x = np.linspace(0,5,no_points)
y= np.linspace(0,6,no_points)
xx, yy = np.meshgrid(x,y)
x_len = x.shape[0]
y_len = y.shape[0]
Z = np.zeros((x_len, y_len))

for p in range(x_len):
	for q in range(y_len):
		data_point = [xx[q][p],yy[q][p]]
		Z[q][p] = post_LDA(data_point,covariance_total,mean_data_0,mean_data_1)
		
plt.contour(xx,yy,Z,1)
plt.title('LDA')
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()
