import numpy as np
import matplotlib.pyplot as plt

data = np.ones((100, 3))
data[:50,0], data[50:,0] = np.random.normal(0, 1, 50), np.random.normal(2, 1, 50)
data[:50,1], data[50:,1] = np.random.normal(0, 1, 50), np.random.normal(2, 1, 50)
target = np.zeros(100)
target[:50], target[50:] = -1 * np.ones(50), np.ones(50)

# Computes the mean squared error of a prediction of w

def mse(x,xhat):
	n = len(x)
	diff = np.subtract(x,xhat)
	squared_sum = np.sum(np.square(diff))
	mse = squared_sum/n
	return mse


# Perceptron updating rule and plotting

size_w = data.shape[1]
w_initial = np.zeros(size_w)  
#print(w_initial.shape)
#print(target)
w = w_initial
w_min = w
s_min = np.dot(np.transpose(target),np.dot(data,w))
#print(w.shape)
for i in range(10):
	for j in range(100):
		#print(j)
		#print(np.sign(np.dot(data,w)))
		#print(target)
		s = np.dot(np.transpose(target),np.sign(np.dot(data,w)))
		if s >= s_min:
			s_min = s
			w_min = w
		predict_j = np.sign(np.dot(np.transpose(w),data[j]))
		if target[j] * predict_j > 0: 
			w = w
		else:
			w = w + target[j]*data[j]

print("This is w_min" + str(w_min))
print("This is w" + str(w))
		
perceptron_slope = -w[0]/w[1]
perceptron_intercept = - w[2]/w[1]
perceptron_estimate = []
for i in data[:,0]:
	estimate = perceptron_intercept + perceptron_slope*i
	perceptron_estimate.append(estimate) 
	plt.plot(i,estimate,'.y-')

s_w = np.dot(target,np.sign(np.dot(data,w)))
s_w_min = np.dot(target,np.sign(np.dot(data,w_min)))
print("This is the sum of w " + str(s_w))
print("This is the sum of w_min " + str(s_w_min))

if s_w_min > s_w:
	new_w = w_min
else:
	new_w = w
print('This is the new w' + str(new_w))

#print(w)
new_perceptron_slope = -new_w[0]/new_w[1]
new_perceptron_intercept = - new_w[2]/new_w[1]
new_perceptron_estimate = []
for i in data[:,0]:
	new_estimate = new_perceptron_intercept + new_perceptron_slope*i
	new_perceptron_estimate.append(new_estimate) 
	plt.plot(i,new_estimate,'.k-')

col = np.where(target<0,'r','b')
plt.scatter(data[:,0],data[:,1],c = col)
plt.title('Improved perceptron')
plt.xlabel('x values')
plt.ylabel('y values')
plt.legend()
plt.show()

