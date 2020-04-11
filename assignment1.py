# Name: Pranesh Shrestha
# Course: ECE-4332
# Assignment: 1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_excel('proj1Dataset.xlsx')
#sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
mean_value = df['Horsepower'].mean(skipna=True)
df.fillna(mean_value, inplace=True)  # fills Nan with mean of the Horsepower

# closed-form solution
df['just ones'] = 1
# Normalizes the data ranging from 0 to 1
weight_norm = df['Weight'].max()
horsepower_norm = df['Horsepower'].max()
df['Weight'] = df[['Weight']] / weight_norm
df['Horsepower'] = df[['Horsepower']] / horsepower_norm

x = np.array(df['Weight'])
X = np.array(df[['Weight', 'just ones']])  # Design Matrix
Y = np.array(df['Horsepower'])  # Target Matrx

weight_matrix = (np.linalg.inv(np.transpose(X)@X)
                 )@np.transpose(X)@Y  # Closed Form Equation
print(weight_matrix.shape)
predict1 = X@weight_matrix  # Prediction
plt.figure(figsize=(10.5, 5.6))
plt.subplot(1, 2, 1)
plt.scatter(x=x*weight_norm, y=Y*horsepower_norm)
plt.plot(x*weight_norm, predict1*horsepower_norm, color='purple')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.title('Closed form')

# Gradient Descent Method

# assumed weight and counter is initialized within function


def gradient_(g_weight=np.array([0.1, 0.2]), counter=0):
    g_weight = g_weight - 0.001 * 2 * \
        (np.transpose(g_weight)@np.transpose(X)@X - np.transpose(Y)@X)
    counter = counter + 1
    if counter == 300:
        return g_weight
    return gradient_(g_weight, counter)


# for calculating weight from gradient descent method (Iterative method)
final_gradient = gradient_()
predict2 = X@final_gradient
plt.subplot(1, 2, 2)
plt.scatter(x=x*weight_norm, y=Y*horsepower_norm)
plt.plot(x*weight_norm, predict2*horsepower_norm, color='purple')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.title('Gradient Descent Method')

plt.show()
