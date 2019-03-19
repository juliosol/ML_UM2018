import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from matplotlib.patches import Ellipse

np.random.seed(1234)

def random_posdef(n):
  A = np.random.rand(n, n)
  return np.dot(A, A.transpose())

# Parameter initialization ###
K = 2
pi = [1.0/K for i in range(K)]
means = [[0,0] for i in range(K)]
#means = [[-0.2,-0.5],[-1.2,-1.6]]
covs = [random_posdef(2) for i in range(K)]

gmm_data = np.load('gmm_data.npy')

################# Scatter plot of data ###########
plt.scatter(gmm_data[:,0],gmm_data[:,1])
plt.show()

'''
############## EM algorithm ##################

## Probability that a given data point came from one of the K clusters 

def em_gmm(data,means,covs,pi,tol, max_iter):
	ll_old = 0.0
	[num_points,dim] = gmm_data.shape

	for l in range(max_iter):
	
	### E step #####
	#### This part is for evaluating the current responsibilities 

		points_classes = np.zeros((K,num_points))
		for i in range(K):
			for j in range(num_points):
				#print(pi[i])
				#print(data[j][0])
				points_classes[i][j] = pi[i] * mvn(means[i],covs[i]).pdf(data[j])
		#print(points_classes[0][500])
		points_classes /= points_classes.sum(0) 
		#print(points_classes[0][100])
		#print(points_classes[0][0])
		
		#### M step ############
		pi = np.zeros(K)
		for i in range(K):
			for j in range(num_points):
				pi[i] = pi[i] + points_classes[i][j]
		pi /= num_points
	
		means = np.zeros((K,dim))
		for i in range(K):
			for j in range(num_points):
				means[i] = means[i] + points_classes[i][j] * data[j]
			means[i] /= points_classes[i][:].sum()
			#print(means)

		covs = np.zeros((K,dim,dim))
		for i in range(K):
			for j in range(num_points):
				ys = np.reshape(data[j] - means[i],(2,1))
				covs[i] = covs[i] + points_classes[i][j] * np.dot(ys,ys.T)
			covs[i] /= points_classes[i][:].sum()
	
		ll_new = 0.0
		for i in range(num_points):
			s = 0
			for j in range(K):
				s = s + pi[j] * mvn(means[j],covs[j]).pdf(data[i][0])
			ll_new = ll_new + np.log(s)
		if np.abs(ll_new - ll_old) < tol:
			return pi, means, covs, ll_new
		ll_old = ll_new
	return pi, means, covs, ll_new

A = em_gmm(gmm_data,means,covs,pi,0.005,100)
print(A[1])
print(A[2])
print(A[3])



def plot_ellipse(pos, cov, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]
    
    if ax is None:
        ax = plt.gca()
    
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    
    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(abs(vals))
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    
    ax.add_artist(ellip)
    return ellip    
    
def show(X, mu, cov):

    plt.cla()
    K = len(mu) # number of clusters
    colors = ['b', 'k', 'g', 'c', 'm', 'y', 'r']
    plt.plot(X.T[0], X.T[1], 'm*')
    for k in range(K):
    	plot_ellipse(mu[k], cov[k],  alpha=0.6, color = colors[k % len(colors)])  
    plt.show()

    
#fig = plt.figure(figsize = (13, 6))
#fig.add_subplot(121)
show(gmm_data, A[1], A[2])
#fig.add_subplot(122)
#plt.plot(np.array(A[3]))
#plt.show()

#print(A)
#print(gmm_data.shape)
'''