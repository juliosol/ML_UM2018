import numpy as np
import matplotlib.pyplot as plt
from math import exp
from scipy.stats import multivariate_normal

n = 100

xtest = np.linspace(-5,5,n).reshape(-1,1)
#print(xtest.shape)


D = np.array([[-1.3,2],[2.4,5.2],[-2.5,-1.5],[-3.3,-0.8],[0.3,0.3]])
D_x_nonflat = np.array([[x] for x in D[:,0]])
D_x = [item for sublist in D_x_nonflat for item in sublist]
D_y_nonflat = np.array([[x for x in D[:,1]]])
D_y = [item for sublist in D_y_nonflat for item in sublist]
#print(D_x)

def kernel(x1,x2,sigma):
	diff = np.subtract(x1, x2)
	diff_squared = diff**2
	return np.exp(diff_squared/(-2 * sigma**2))

def covariance_matrix_generator(covariance_input_vector_x,sigma):
	total_n = covariance_input_vector_x.shape[0]
	total_cov_matrix = np.eye(total_n)
	for i in range(total_n):
		for j in range(total_n):
			total_cov_matrix[i][j] = kernel(covariance_input_vector_x[i],covariance_input_vector_x[j],sigma)
	total_cov_matrix = total_cov_matrix + 0.15e-10*np.eye(total_n)
	return total_cov_matrix

def sub_sigma_generator(covariance_matrix,a,b):
	sigma_a = covariance_matrix[np.ix_(a,a)]
	sigma_ab = covariance_matrix[np.ix_(a,b)]
	sigma_ba = covariance_matrix[np.ix_(b,a)]
	sigma_b = covariance_matrix[np.ix_(b,b)]
	return sigma_a,sigma_ab,sigma_ba,sigma_b

def sigma_a_cond_b(sigma_a,sigma_ab,sigma_ba,sigma_b):
	product_sigma = np.dot(sigma_ab,np.dot(np.linalg.inv(sigma_b),sigma_ba))
	sigma_a_cond_b = np.subtract(sigma_a,product_sigma)
	return sigma_a_cond_b

def mean_a_cond_b(mu_a,mu_b,sigma_ab,sigma_b,known_vector_x_b):
	size_a = len(known_vector_x_b)
	product = np.dot(sigma_ab,np.dot(np.linalg.inv(sigma_b),np.subtract(np.transpose(known_vector_x_b),np.zeros(size_a)*0)))
	mean_a_b = np.add(mu_a, product)
	return mean_a_b

def gaussian_sampler(covariance_matrix,no_samples,mean):
	ys = np.random.multivariate_normal(mean,covariance_matrix,no_samples)
	plt.plot(xtest,np.transpose(ys))
	plt.plot(D_x,D_y,'ro')
	plt.show()	
	return 

def cond_gaussian_sampler(covariance_matrix,no_samples, x_values):
	
	ys = np.random.multivariate_normal(xtest,mean,covariance_matrix)
	plt.plot(xtest,np.transpose(ys))
	plt.scatter(mean)
	plt.plot(D_x,D_y,'ro')
	plt.show()	
	return 

def rbf_kernel(x1, x2, sigma):
    return exp(-1 * ((x1-x2) ** 2) / (2*sigma**2))

def gram_matrix(xs,sigma):
    return [[rbf_kernel(x1,x2,sigma) for x2 in xs] for x1 in xs]

#prueba = gram_matrix([-1.3,2.4,-2.5,-3.3,0.3])
#print(prueba)

def main():
	
	############################
	## Problem 4 a) part 1 and 2

	# Case 1
	n = xtest.shape[0]
	mean_p1 = np.zeros(n)
	sigma_p1_1 = 0.3
	covariance_matrix_p1_1 = covariance_matrix_generator(xtest,sigma_p1_1)
	prior_functions_p1_1 = gaussian_sampler(covariance_matrix_p1_1,5,mean_p1)
	
	# Case 2
	sigma_p1_2 = 0.5
	covariance_matrix_p1_2 = covariance_matrix_generator(xtest,sigma_p1_2)
	prior_functions_p1_2 = gaussian_sampler(covariance_matrix_p1_1,5,mean_p1)

	# Case 3

	sigma_p1_3 = 1
	covariance_matrix_p1_3 = covariance_matrix_generator(xtest,sigma_p1_3)
	prior_functions_p1_3 = gaussian_sampler(covariance_matrix_p1_1,5,mean_p1)

	###############################
	####### Problem 4b) part 1 and 2
	unkown_length = xtest.shape[0]
	known_length = len(D_x)
	total_x = np.concatenate([xtest,D_x_nonflat])
	total_n = total_x.shape[0]
	
	a = np.array(range(unkown_length))
	b = np.array([100,101,102,103,104])
	
	mu_a = np.zeros(unkown_length)
	mu_b = np.zeros(known_length)
	
	sigma_1 = 0.3
	sigma_2 = 0.5
	sigma_3 = 1.0

	# Case 1
	covariance_matrix_1 = covariance_matrix_generator(total_x,sigma_1)
	#covariance_matrix_1 = gram_matrix(total_x, sigma_1)
	[sigma_a_1,sigma_ab_1,sigma_ba_1,sigma_b_1] = sub_sigma_generator(np.array(covariance_matrix_1),a,b)
	print(sigma_b_1)

	sigma_a_b_1 = sigma_a_cond_b(sigma_a_1,sigma_ab_1,sigma_ba_1,sigma_b_1)
	
	mu_a_b_1 = mean_a_cond_b(mu_a,mu_b,sigma_ab_1,sigma_b_1,D_x)

	#puntos = [-1.3,2.4,-2.5,-3.3,0.3]
	#prueba_bb = gram_matrix([-1.3,2.4,-2.5,-3.3,0.3])

	#mu_a_b_1 = np.zeros(100) + np.dot(sigma_ab_1,np.dot(np.linalg.inv(prueba_bb),np.array(puntos)))
	#print(mu_a_b_1)
	
	post_functions = gaussian_sampler(sigma_a_b_1,5,mu_a_b_1)
	
	xs = np.arange(-1, 1, 0.01)
	mean = [0 for x in xtest]
	#gram = gram_matrix(xtest)
	#print(np.linalg.det(gram))

	plt_vals = []
	for i in range(0, 5):
		ys = np.random.multivariate_normal(mu_a_b_1, sigma_a_b_1)
		plt_vals.extend([xtest, ys, "k"])
	plt.plot(*plt_vals)
	plt.plot(D_x,D_y,'ro')
	plt.show()



	# Case 2
	covariance_matrix_2 = covariance_matrix_generator(total_x,sigma_2)
	[sigma_a_2,sigma_ab_2,sigma_ba_2,sigma_b_2] = sub_sigma_generator(covariance_matrix_2,a,b)

	sigma_a_b_2 = sigma_a_cond_b(sigma_a_2,sigma_ab_2,sigma_ba_2,sigma_b_2)
	
	mu_a_b_2 = mean_a_cond_b(mu_a,mu_b,sigma_ab_2,sigma_b_2,D_x)
	
	post_functions = gaussian_sampler(sigma_a_b_2,5,mu_a_b_2)

	# Case 3

	covariance_matrix_3 = covariance_matrix_generator(total_x,sigma_3)
	[sigma_a_3,sigma_ab_3,sigma_ba_3,sigma_b_3] = sub_sigma_generator(covariance_matrix_3,a,b)

	sigma_a_b_3 = sigma_a_cond_b(sigma_a_3,sigma_ab_3,sigma_ba_3,sigma_b_3)
	mu_a_b_3 = mean_a_cond_b(mu_a,mu_b,sigma_ab_3,sigma_b_3,D_x)
	#print(mu_a_b_1.shape)
	
	post_functions = gaussian_sampler(sigma_a_b_3,5,mu_a_b_3)

main()
