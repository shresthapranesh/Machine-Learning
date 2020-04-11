# Name: Pranesh Shrestha
# Assignment: 3 (ECE-4332)

import numpy as np
import matplotlib.pyplot as plt
from statistics import mean


class nonLinearModel:
    def __init__(self, L, N):
        self.L = L
        self.N = N
        self.dataset = []
        self.h_x = np.sin(2*3.1415*np.random.uniform(0, 1, N)
                          )  # pure Sine wave
        # 100 Datasets
        for i in range(0, L):
            self.X = np.random.uniform(0, 1, N)
            self.t = np.sin(2*3.1415*self.X) + np.random.normal(0, 0.3, N)
            self.dataset.append([self.X, self.t])
        self.degree = 25
        self.N_lambda = 50

    def regression_regularization(self, design_matrix, t, lamb):
        weight = np.dot(np.dot(np.linalg.inv(
            np.dot(np.transpose(design_matrix), design_matrix) + lamb * np.identity(self.degree+1)), np.transpose(design_matrix)), t)
        return weight

    def rbf_gen(self, x_, N, degree):

        mean1 = np.linspace(0, 1, degree+1)
        design_matrix = np.empty((x_.shape[0], degree+1), dtype=float)
        for i in range(x_.shape[0]):
            for j in range(degree+1):
                if j == 0:
                    design_matrix[i][j] = 1
                else:
                    design_matrix[i][j] = np.exp(
                        -((x_[i]-mean1[j])**2)/(2*(0.1**2)))
        return design_matrix

    def test_error(self, weights):
        X_test = np.random.uniform(0, 1, 1000)
        t_test = np.sin(2*3.1415*X_test) + np.random.normal(0, 0.3, 1000)
        design_matrix = self.rbf_gen(X_test, N=1000, degree=self.degree)

        rms_list = []
        for weight in weights:
            expected_value = np.dot(design_matrix, weight)
            cost_function = expected_value - t_test
            cost_function = np.square(cost_function)
            rms = np.sqrt(np.sum(cost_function)/1000)
            rms_list.append(rms)

        return rms_list

    def bias_variance(self, dataset_expected_value):

        expected_values = []
        for j in range(self.N_lambda):
            temp1 = dataset_expected_value[j]
            temp2 = []
            for k in range(self.N):
                temp2.append(mean([float(i[k]) for i in temp1]))
            expected_values.append(temp2)

        bias = []
        variance = []

        for i in range(self.N_lambda):
            temp = expected_values[i]
            temp2 = []
            for j in range(self.N):
                temp2.append(temp[j]-self.h_x[j])
            bias.append(mean(temp2))

        for i in range(self.N_lambda):
            temp1 = dataset_expected_value[i]
            temp2 = expected_values[i]
            temp5 = []
            for j in range(self.N):
                temp3 = [k[j] for k in temp1]
                temp4 = [(a - temp2[j])**2 for a in temp3]
                temp5.append(mean(temp4))

            variance.append(mean(temp5))
        return bias, variance

    def func_fit(self):
        X_train = [i[0] for i in self.dataset]
        y_train = [i[1] for i in self.dataset]

        weights = []
        dataset_expected_value = []

        for lammb in np.linspace(-1, 3, self.N_lambda):
            lamb = np.exp(lammb)
            temp_expected_value = []
            errs = []
            w = []
            for k in range(self.L):
                design_matrix = self.rbf_gen(
                    X_train[k], N=self.N, degree=self.degree)
                w_ = self.regression_regularization(
                    design_matrix, y_train[k], lamb)
                w.append(w_)
                temp_expected_value.append(
                    np.dot(design_matrix, np.array(w_)))
            #print([a[0] for a in w])
            temp2 = []
            for i in range(w_.shape[0]):
                temp2.append(mean([float(a[i]) for a in w]))

            weights.append(temp2)
            dataset_expected_value.append(temp_expected_value)
        bias_sq, variance = self.bias_variance(dataset_expected_value)
        test_errs = self.test_error(weights)
        return test_errs, bias_sq, variance


mymodel = nonLinearModel(L=100, N=25)
test_err, bias_sq, variance = mymodel.func_fit()


bias_variance = []
for i in range(mymodel.N_lambda):
    bias_variance.append((bias_sq[i] + variance[i])/2)

m = np.linspace(-1, 4, mymodel.N_lambda)
# plotting for training N = 100
plt.figure(figsize=(10.5, 5.6))
plt.plot(m, test_err, label='test error')
plt.plot(m, variance, label='variance')
plt.plot(m, bias_sq, label=r'$bias^2 $')
plt.plot(m, bias_variance, label=r'$bias^2 + variance$')
plt.xlabel(r'$ln\lambda$')
plt.title('Training data N = 25')
plt.legend()


plt.show()
