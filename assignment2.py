# Name: Pranesh Shrestha
# Course: ECE - 4332 (Machine Learning)
# Assignment: 2

import numpy as np
import matplotlib.pyplot as plt


class train_test_data():
    def __init__(self, N):
        self.N = N
        self.X_ = np.random.uniform(0, 1, self.N)
        self.y_ = np.sin(2*3.1415*self.X_) + \
            np.random.normal(0, 0.3, self.N)

    def calculate_Erms(self):
        design_matrices = []
        for i in range(0, 10):
            basis = np.array([])
            for j in self.X_:
                basis = np.append(basis, j ** i)
            if i == 0:
                design_matrix = np.array(basis).reshape(self.N, 1)
            else:
                design_matrix = np.append(
                    design_matrix, basis.reshape(self.N, 1), axis=1)
            design_matrices.append(design_matrix)

        weights = []
        for matrix in design_matrices:
            weight = np.dot(np.dot(np.linalg.inv(
                np.dot(np.transpose(matrix), matrix)), np.transpose(matrix)), self.y_)  # no more @ (:
            weights.append(weight)
        Erms = []
        for index in range(10):
            first_term = np.dot(np.dot(np.transpose(weights[index]), np.transpose(
                design_matrices[index])), np.dot(design_matrices[index], weights[index]))
            second_term = 2 * \
                (np.dot(np.transpose(self.y_), np.dot(
                    design_matrices[index], weights[index])))
            third_term = np.dot(np.transpose(self.y_), self.y_)
            cost_function = first_term - second_term + third_term
            rms = np.sqrt(cost_function/self.N)
            Erms.append(rms)
        return Erms


# for N = 10 for training data
# np.random.seed(101)
train_N_10 = train_test_data(10)
train_err_10 = train_N_10.calculate_Erms()

test_N_100 = train_test_data(100)
test_err_100 = test_N_100.calculate_Erms()

# for N = 100 for training data

train_N_100 = train_test_data(100)
train_err_100 = train_N_100.calculate_Erms()

test_NN_100 = train_test_data(100)
test_Nerr_100 = test_NN_100.calculate_Erms()

m = range(0, 10)

# plotting for training N = 10
plt.figure(figsize=(10.5, 5.6))
plt.subplot(1, 2, 1)
plt.plot(m, train_err_10, marker='o', label='training set')
plt.plot(m, test_err_100, marker='o', label='test set')
plt.xlabel('M')
plt.ylabel('E rms')
plt.title('Training data N = 10')
plt.legend()

# plotting for training N = 100
plt.subplot(1, 2, 2)
plt.plot(m, train_err_100, marker='o', label='training set')
plt.plot(m, test_Nerr_100, marker='o', label='test set')
plt.xlabel('M')
plt.ylabel('E rms')
plt.title('Training data N = 100')
plt.legend()

plt.show()
