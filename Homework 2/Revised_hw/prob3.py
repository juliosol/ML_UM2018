import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal 

# we defined a class for sequential bayesian learner
class bayesian_linear_regression(object):

    # initialized with covariance matrix(sigma), mean vector(mu) and prior(beta)
    def __init__(self,sigma,mu,beta):
        self.sigma = sigma
        self.mu = mu
        self.beta = beta

    # you need to implement the update function
    # when received additional design matrix phi and continuous label t
    def update(self,phi,t):
        #print("This is the starting sigma" + str(self.sigma))
        #print("This is the starting mu" + str(self.mu))
        sigma = np.linalg.inv(np.linalg.inv(self.sigma) + self.beta*np.dot(np.transpose(phi),phi)) 
        mu = np.dot(sigma,(self.beta*np.dot(np.transpose(phi),t) + np.dot(np.linalg.inv(self.sigma),self.mu))) 
        self.sigma = sigma
        self.mu = mu
        #print("This is the new sigma" + str(self.sigma))
        #print("This is the new mu" + str(self.mu))     
        return self.sigma,self.mu

def multivariate_gaussian_pdf(x,sigma,mu):
    d = mu.shape[0]
    diff_mean = np.subtract(x,mu)
    product_mat = np.dot(np.transpose(diff_mean),np.dot(np.linalg.inv(sigma),diff_mean))
    exponential = np.exp(-1/2*produt_mat)
    sigma_det = np.linalg.inv(sigma)
    pdf = exponential/(np.sqrt((2*math.pi)**D*det_sigma))
    return pdf


def multivariate_gaussian_plot(mu,sigma):
    # create 2 kernels
    m1 = mu
    s1 = sigma
    k1 = multivariate_normal(mean=m1, cov=s1)

    # create a grid of (x,y) coordinates at which to evaluate the kernels
    xlim = (-3, 3)
    ylim = (-3, 3)
    xres = 100
    yres = 100

    x = np.linspace(xlim[0], xlim[1], xres)
    y = np.linspace(ylim[0], ylim[1], yres)
    xx, yy = np.meshgrid(x,y)

    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    zz = k1.pdf(xxyy)

    # reshape and plot image
    img = zz.reshape((xres,yres))
    plt.imshow(img); plt.show()


def data_generator(size,scale):
    x = np.random.uniform(low=-3, high=3, size=size)
    rand = np.random.normal(0, scale=scale, size=size)
    y = 0.5 * x - 0.3 + rand
    phi = np.array([[x[i], 1] for i in range(x.shape[0])])
    t = y
    return phi, t


def main():
    # initialization
    alpha = 2
    sigma_0 = np.diag(1.0/alpha*np.ones([2]))
    mu_0 = np.zeros([2])
    beta = 1.0
    blr_learner = bayesian_linear_regression(sigma_0, mu_0, beta=beta)
    multivariate_gaussian_plot(blr_learner.mu,blr_learner.sigma)

    num_episodes = 20

    for epi in range(num_episodes):
        phi, t = data_generator(1,1.0/beta)
        blr_learner.update(phi,t)
        print("This is the mean for initial distribution " + str(blr_learner.mu))
        print("This is the sigma for initial distribution " + str(blr_learner.sigma))
        if epi == 0:
            multivariate_gaussian_plot(blr_learner.mu,blr_learner.sigma)
            print("This is the mean for distribution after 1 instance " + str(blr_learner.mu))
            print("This is the sigma for initial distribution " + str(blr_learner.sigma))
        if epi == 9:
            multivariate_gaussian_plot(blr_learner.mu,blr_learner.sigma)
            print("This is the mean for distribution after 10 instances " + str(blr_learner.mu))
            print("This is the sigma for initial distribution " + str(blr_learner.sigma))
        if epi == 19:
            multivariate_gaussian_plot(blr_learner.mu,blr_learner.sigma)
            print("This is the mean for distribution after 20 instance " + str(blr_learner.mu))
            print("This is the sigma for initial distribution " + str(blr_learner.sigma))

main()


