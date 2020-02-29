import numpy as np
import matplotlib.pyplot as plt
from statistics import mean


class nonLinearModel:
    def __init__(self, L, N):
        self.L = L
        self.N = N
        self.dataset = []
        np.random.seed(10)
        self.h_x = np.sin(2*3.1415*np.random.uniform(0, 1, N))
        for i in range(0, L):
            np.random.seed(10)
            self.X = np.random.uniform(0, 1, N)
            np.random.seed(10)
            self.t = np.sin(2*3.1415*self.X) + np.random.normal(0, 0.3, N)
            self.dataset.append([self.X, self.t])
        self.degree = 1

    def regression_regularization(self, design_matrix, t, lamb):
        weight = np.dot(np.dot(np.linalg.inv(
            np.dot(np.transpose(design_matrix), design_matrix) + lamb * np.identity(self.degree+1)), np.transpose(design_matrix)), t)
        return weight

    def rbf_gen(self, x_):
        mean1 = mean([float(i) for i in x_])
        design_matrix = np.empty((self.N, self.degree+1), dtype=float)
        for i in range(self.N):
            for j in range(self.degree+1):
                if j == 0:
                    design_matrix[i][j] = 1
                else:
                    design_matrix[i][j] = np.exp(
                        -((x_[i]-mean1)**2)/(2*(0.1**2)))
        return design_matrix

    def test_error(self, design_matrix, weight, t):
        first_term = np.dot(np.dot(np.transpose(weight), np.transpose(
            design_matrix)), np.dot(design_matrix, weight))
        second_term = 2 * \
            (np.dot(np.transpose(t), np.dot(
                design_matrix, weight)))
        third_term = np.dot(np.transpose(t), t)
        cost_function = first_term - second_term + third_term
        rms = np.sqrt(cost_function/100)

        return rms

    def bias_variance(self, dataset_expected_value):

        expected_values = []
        for j in range(10):
            temp1 = dataset_expected_value[j]
            temp2 = []
            for k in range(self.N):
                temp2.append(mean([float(i[k]) for i in temp1]))
            expected_values.append(temp2)

        bias = []
        variance = []

        for i in range(10):
            temp = expected_values[i]
            temp2 = []
            for j in range(self.N):
                temp2.append(temp[j]-self.h_x[j])
            bias.append(mean(temp2))

        for i in range(10):
            temp1 = dataset_expected_value[i]
            temp2 = expected_values[i]
            temp5 = []
            for j in range(self.N):
                temp3 = [k[j] for k in temp1]
                temp4 = [(a - temp2[j])**2 for a in temp3]
                temp5.append(mean(temp4))

            variance.append(mean(temp5))
        print(len(variance))
        return bias, variance

    def func_fit(self):
        X_train = [i[0] for i in self.dataset]
        y_train = [i[1] for i in self.dataset]

        test_errs = []
        w = []
        dataset_expected_value = []

        for lammb in np.linspace(-3, 2, 10):
            lamb = np.exp(lammb)
            temp_expected_value = []
            errs = []
            for k in range(self.L):
                design_matrix = self.rbf_gen(X_train[k])
                w_ = self.regression_regularization(
                    design_matrix, y_train[k], lamb)
                temp_expected_value.append(
                    np.dot(design_matrix, np.array(w_)))
                errs.append(self.test_error(design_matrix, w_, y_train[k]))

            dataset_expected_value.append(temp_expected_value)
            test_errs.append(mean(errs))

        bias_sq, variance = self.bias_variance(dataset_expected_value)
        return test_errs, bias_sq, variance


mymodel = nonLinearModel(L=100, N=25)

test_err, bias_sq, variance = mymodel.func_fit()

bias_variance = []
for i in range(10):
    bias_variance.append((bias_sq[i]+variance[i])/2)

m = np.linspace(-3, 2, 10)
# plotting for training N = 100
plt.figure(figsize=(10.5, 5.6))
#plt.plot(m, test_err, label='test error')
plt.plot(m, variance, label='variance')
plt.plot(m, bias_sq, label=r'$bias^2 $')
plt.plot(m, bias_variance, label=r'$bias^2 + variance$')
plt.xlabel(r'$ln\lambda$')
plt.title('Training data N = 25')
plt.legend()


plt.show()
