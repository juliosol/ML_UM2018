import numpy as np
import collections

# load the training data 
train_features = np.load("spam_train_features.npy")
train_labels = np.load("spam_train_labels.npy")


# load the test data 
test_features = np.load("spam_test_features.npy")
test_labels = np.load("spam_test_labels.npy")

print(test_labels[1])
print(test_labels[2])