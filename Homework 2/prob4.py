import numpy as np
import matplotlib.pyplot as plt



n = 100

xtest = np.linspace(-5,5,n).reshape(-1,1)
print(xtest)

def kernel(x1,x2,sigma):
	diff = x1 - x2
	diff_squared = np.dot(np.transpose(x1-x2),(x1-x2))
	return np.exp(diff_squared/(-2 * sigma**2))


def main():
	
	############################
	## Problem 4 a) part 1 and 2

	sigma = 0.3
	cov_matrix = np.eye(100)
	for i in range(n):
		for j in range(n):
			cov_matrix[i][j] = kernel(xtest[i],xtest[j],sigma)


	#kernel(xtest,xtest,sigma)
	print(cov_matrix)
	#n = cov_matrix.shape[0]

	cholesky_cov_matrix = np.linalg.cholesky(cov_matrix + 1e-15*np.eye(100))
	#u = np.random.multivariate_normal(np.zeros(n), np.identity(n))
	f_prior = np.dot(cholesky_cov_matrix,np.random.normal(size = (100,5)))
	plt.plot(xtest,f_prior)
	plt.show()

	## Problem 4b)

main()
