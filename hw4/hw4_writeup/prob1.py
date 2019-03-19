import numpy as np 
from sklearn import svm

np.random.seed(3)

mean_1 = [ 2.0 , 0.2 ]
cov_1 = [ [ 1 , .5 ] , [ .5 , 2.0 ]]

mean_2 = [ 0.4 , -2.0 ]
cov_2 = [ [ 1.25 , -0.2 ] , [ -0.2, 1.75 ] ]

x_1 , y_1 = np.random.multivariate_normal( mean_1 , cov_1, 15).T
x_2 , y_2 = np.random.multivariate_normal( mean_2 , cov_2, 15).T

import matplotlib.pyplot as plt 

plt.plot( x_1 , y_1 , 'x' )
plt.plot( x_2 , y_2 , 'ro')
plt.axis('equal')
#plt.show()

X = np.zeros((30,2))
X[0:15,0] = x_1
X[0:15,1] = y_1
X[15:,0] = x_2
X[15:,1] = y_2

y = np.zeros(30)
y[0:15] = np.ones(15)
y[15:] = -1 * np.ones(15)


# Function to create the meshgrid to plot the data.

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


# Function to plot contours of classifier
def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


## This is the code for the linear SVM

param_C = 100

clf = svm.SVC(C = param_C, kernel='linear')
model1 = clf.fit(X, y)

support_vectors = model1.support_vectors_
#number_support_vectors = model1.n_support_vectors_
#print('The support vectors for the linear kernel are ' + str(support_vectors))
print('The number of suppoert vectors in linear kernel with current C is ' + str(len(support_vectors)))

w = model1.coef_[0]
a = - w[0]/w[1]
xx = np.linspace(np.amin(np.array(X)),np.amax(np.array(X)) + 1)
yy = a * xx - clf.intercept_[0] / w[1]
line_0 = plt.plot(xx,yy,'k-')

title = 'Linear kernel  with C = '

plt.scatter(X[:,0],X[:,1],c = y)
plt.title(title + str(param_C))
plt.show()

## THis is the code for the RBF kernel SVM

param_C_rbf = 3

clf_rbf = svm.SVC(C = param_C_rbf, kernel = 'rbf')
model2 = clf_rbf.fit(X,y)

support_vectors_rbf = model2.support_vectors_
#number_suppoert_vectors_rbf = model2.n_support_vectors_
#print('The support vectors for the linear kernel are ' + str(support_vectors_rbf))
print('The number of suppoert vectors is in rbf with current C is ' + str(len(support_vectors_rbf)))


X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
title_2 = 'SVC with rbf kernel with C = '

fig1,sub1 = plt.subplots(1,1)
ax = sub1
plot_contours(ax,model2, xx,yy, cmap = plt.cm.coolwarm, alpha = 0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
#ax.set_xlim(xx.min(), xx.max())
#ax.set_ylim(yy.min(), yy.max())
#ax.set_xlabel('Sepal length')
#ax.set_ylabel('Sepal width')
#ax.set_xticks(())
#ax.set_yticks(())
ax.set_title(title_2 + str(param_C_rbf))
plt.show()
