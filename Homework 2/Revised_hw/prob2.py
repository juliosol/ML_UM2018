import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import *
from scipy.stats import multivariate_normal


# feel free to read the two examples below, try to understand them
# in this problem, we require you to generate contour plots

# generate contour plot for function z = x^2 + 2*y^2
def plot_contour():

    plot_delta = 0.025
    plot_x = np.arange(-3.0, 3.0, plot_delta)
    plot_y = np.arange(-3.0, 3.0, plot_delta)
    X, Y = np.meshgrid(plot_x, plot_y)
    x_len = plot_x.shape[0]
    y_len = plot_y.shape[0]
    Z = np.zeros((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):
            Z[j][i] = X[j][i] ** 2 + 2 * (Y[j][i] ** 2)

    plt.clf()
    plt.axis("square")
    cs = plt.contour(X, Y, Z)
    plt.clabel(cs, inline=0.1, fontsize=10)
    plt.show()


# generate heat plot (image-like) for function z = x^2 + 2*y^2
def plot_heat():
    plot_delta = 0.025
    plot_x = np.arange(-3.0, 3.0, plot_delta)
    plot_y = np.arange(-3.0, 3.0, plot_delta)
    X, Y = np.meshgrid(plot_x, plot_y)
    x_len = plot_x.shape[0]
    y_len = plot_y.shape[0]
    Z = np.zeros((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):
            Z[j][i] = X[j][i] ** 2 + 2 * (Y[j][i] ** 2)

    plt.clf()
    img = Z
    plt.imshow(img, interpolation='none', extent=[-3.0, 3.0, -3.0, 3.0],cmap="plasma")
    plt.colorbar()
    plt.axis("square")
    plt.show()



# This function receives the parameters of a multivariate Gaussian distribution
# over variables x_1, x_2 .... x_n as input and compute the marginal
#
def marginal_for_gaussian(sigma,mu,given_indices):
    # given selected indices, compute marginal distribution for them
    
    marginal_mu = []
    marginal_sigma = sigma[np.ix_(given_indices,given_indices)]
    
    for i in given_indices:
        marginal_mu.append(mu[i])

    x = np.random.multivariate_normal(marginal_mu, marginal_sigma) 
    return marginal_mu, marginal_sigma


def conditional_for_gaussian(sigma,mu,given_indices,given_values):
    # given some indices that have fixed value, compute the conditional distribution
    # for rest indices
    n = len(mu)
    b = given_indices
    total_indices = list(range(n))
    a = list(set(total_indices) - set(b))

    sigma_a = sigma[np.ix_(a,a)]
    sigma_ab = sigma[np.ix_(a,b)]
    sigma_ba = sigma[np.ix_(b,a)]
    sigma_b = sigma[np.ix_(b,b)]

    #print(sigma_a)
    #print(sigma_ab)
    #print(sigma_ba)
    #print(sigma_b)

    mu_a = []
    mu_b = []

    for i in a:
        mu_a.append(mu[i])

    for i in b:
        mu_b.append(mu[i])

    #print("This is the mean of the given indices" + str(mu_a))

    sigma_a_cond_b = sigma_a - np.dot(np.dot(sigma_ab,np.linalg.inv(sigma_b)),sigma_ba)
    #print("This is the sigma part computation of conditional mean" + str(np.array(sigma_ab*np.linalg.inv(sigma_b)*(np.array(given_values) - np.array(mu_b)))))
    #print((np.linalg.inv(sigma_b)*(np.array(given_values) - np.array(mu_b))).shape)
    mu_a_cond_b = mu_a + np.dot(np.dot(sigma_ab,np.linalg.inv(sigma_b)),(np.array(given_values) - np.array(mu_b)))

    x = np.random.multivariate_normal(mu_a_cond_b, sigma_a_cond_b)

    return sigma_a_cond_b, mu_a_cond_b

def marginal_gaussian_plot_contour(sigma,mu):
    D = sigma.shape[0]
    k1 = multivariate_normal(mu, sigma)
    # create a grid of (x,y) coordinates at which to evaluate the kernels
    if D == 2:
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
    elif D == 1:
        xlim = (-3, 3)
        xres = 100
        
        x = np.linspace(xlim[0], xlim[1], xres)
        xx = np.meshgrid(x)

        # evaluate kernels at grid points
        #xxyy = np.c_[xx.ravel()]
        zz = k1.pdf(np.transpose(xx))

        # reshape and plot image
        img = zz.reshape(xres)
        plt.plot(x,img); plt.show()
    

test_sigma_1 = np.array(
    [[1.0, 0.5],
     [0.5, 1.0]]
)

test_mu_1 = np.array(
    [0.0, 0.0]
)

test_sigma_2 = np.array(
    [[1.0, 0.5, 0.0, 0.0],
     [0.5, 1.0, 0.0, 1.5],
     [0.0, 0.0, 2.0, 0.0],
     [0.0, 1.5, 0.0, 4.0]]
)

test_mu_2 = np.array(
    [0.5, 0.0, -0.5, 0.0]
)

indices_1 = np.array([0])

indices_2 = np.array([1,2])
values_2 = np.array([0.1,-0.2])

plot_contour()
plot_heat()

[marginal_mu,marginal_sigma] = marginal_for_gaussian(test_sigma_1, test_mu_1, indices_1)
marginal_gaussian_plot_contour(marginal_sigma,marginal_mu)
print("This is the marginal sigma " + str(marginal_sigma))
print("This is the marginal mu " + str(marginal_mu))

[sigma_a_cond_b,mu_a_cond_b] =conditional_for_gaussian(test_sigma_2, test_mu_2, indices_2, values_2)
marginal_gaussian_plot_contour(sigma_a_cond_b,mu_a_cond_b)
print("This is the conditional sigma " + str(sigma_a_cond_b))
print("This is the conditional mu " + str(mu_a_cond_b))
