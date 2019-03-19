import numpy as np
import matplotlib.pyplot as plt

data = np.zeros((100, 3))
val = np.random.uniform(0, 2, 100)
diff = np.random.uniform(-1, 1, 100)
data[:,0], data[:,1], data[:,2] = val - diff, val + diff, np.ones(100)
target = np.asarray(val > 1, dtype = int) * 2 - 1

# Perceptron updating rule and plotting

size_w = data.shape[1]
w_initial = np.zeros(size_w)  
#print(w_initial.shape)
#print(target)
w = w_initial
for i in range(10):
	for j in range(100):
		predict_j = np.sign(np.dot(np.transpose(w),data[j]))
		if target[j] * np.dot(np.transpose(w),data[j]) > 0:
			w = w
		else:
			w = w + target[j]*data[j]

#print(w)
perceptron_slope = -w[0]/w[1]
perceptron_intercept = - w[2]/w[1]
perceptron_estimate = []
for i in data[:,0]:
	estimate = perceptron_intercept + perceptron_slope*i
	perceptron_estimate.append(estimate) 
	plt.plot(i,estimate,'.y-')
#print(data[:,0].shape)
#print(len(perceptron_estimate))

col = np.where(target<0,'r','b')
plt.scatter(data[:,0],data[:,1],c = col)
plt.title('Perceptron classifying')
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()

