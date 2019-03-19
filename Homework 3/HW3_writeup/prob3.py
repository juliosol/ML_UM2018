import numpy as np
import matplotlib.pyplot as plt
import collections

np.random.seed(17)
# data and target are given below 
# data is a numpy array consisting of 100 2-dimensional points
# target is a numpy array consisting of 100 values of 1 or -1
data = np.ones((100, 2))
print(data[1])
data[:,0] = np.random.uniform(-1.5, 1.5, 100)
data[:,1] = np.random.uniform(-2, 2, 100)
z = data[:,0] ** 2 + ( data[:,1] - (data[:,0] ** 2) ** 0.333 ) ** 2  
target = np.asarray( z > 1.5, dtype = int) * 2 - 1

# THis chunk of the data is for dividing the data into colors

# Blue data points happen if target label is -1 and red data points happen d
# if target label is 1.

data_x = data[:,0]
data_y = data[:,1]

neg_one_position = [i for i in range(len(target)) if target[i] == -1]
pos_one_position = [i for i in range(len(target)) if target[i] == 1]

count_neg_one = np.count_nonzero(target == -1)
count_pos_one = np.count_nonzero(target == 1)

#print(neg_one_position)
#print(pos_one_position)

#data_red = np.ones((len(pos_one_position),2))
data_red_x = []
data_red_y = []
#data_blue = np.ones((len(neg_one_position),2))
data_blue_x = []
data_blue_y = []

for j in pos_one_position:
	data_red_x.append(data[j,0])
	data_red_y.append(data[j,1])

for j in neg_one_position:
	data_blue_x.append(data[j,0])
	data_blue_y.append(data[j,1])


plt.scatter(data_red_x, data_red_y, color = 'red')
plt.scatter(data_blue_x,data_blue_y,color = 'blue')


## This part is for implementing the kernel perceptron

def gaussian_kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

size_alpha = data.shape[0]
alpha_initial = np.zeros(size_alpha)  
loss = 0
sigma = 1
alpha = alpha_initial
i = 0
for k in range(10):
	i = 0
	for s in range(100):
		sum_prod_kernel = 0
		for j in range(100):
			sum_prod_kernel = alpha[j] * target[j] * gaussian_kernel(data[j],data[i],sigma) + sum_prod_kernel
		predict_i = np.sign(sum_prod_kernel)
		if target[i] * predict_i > 0:
			alpha[i] = alpha[i]
			loss = loss
		else:
			alpha[i] = alpha[i] + 1
			loss = loss + 1
		i = i + 1


#prediction
sum_product = 0
for j in range(100):
	sum_product = alpha[j]*target[j]*gaussian_kernel(data[j],data[1],sigma) + sum_product
prediction = np.sign(sum_product)

print(data[1])
print(target[1])
print(prediction)



no_points = 100
x = np.linspace(-1.5,1.5,no_points)
y= np.linspace(-2,2,no_points)
xx, yy = np.meshgrid(x,y)
x_len = x.shape[0]
#print(x_len)
y_len = y.shape[0]
Z = np.zeros((x_len, y_len))
#data_function[:,0] = x
#data_function[:,1] = y
#level_zero_contour_x = []
#level_zero_contour_y = []
#function_ker_perc = np.zeros(no_points)
#print(x_len)
#print(y_len)

for p in range(x_len):
	for q in range(y_len):
		sum_product = 0
		data_function = [xx[q][p],yy[q][p]]
		for j in range(100):
			sum_product = alpha[j]*target[j]*gaussian_kernel(data[j],data_function,sigma) + sum_product
		Z[q][p] = np.sign(sum_product)
		if Z[q][p] == -1:
			break

		
print(Z)

plt.contour(xx,yy,Z,0)
plt.show()

'''
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

#print(level_zero_contour_y)
#print(level_zero_contour_x)
#print(function_ker_perc)

#plt.plot(level_zero_contour_x,level_zero_contour_y)
#plt.show()
#print(np.meshgrid(x,y))

#for l in range(no_points):
#	sum_product = 0
#	for j in range(100):
#		sum_product = alpha[j]*target[j]*gaussian_kernel(data[j],data_function[l],sigma) + sum_product
#	function_ker_perc[l] = np.sign(sum_product)
#	if sum_product < 0.000002:
#		level_zero_contour_x.append(data_function[l])
#		level_zero_contour_y.append(data_function[l])
'''
