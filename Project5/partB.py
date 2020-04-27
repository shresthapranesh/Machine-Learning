import numpy as np
import matplotlib.pyplot as plt
from feedforward import FullyConnectedRegressor

rng = np.random.default_rng(seed=100)

X_train = 2 * rng.uniform(size=(1, 50))-1
t = np.sin(2*3.1415*X_train) + 0.3*rng.normal(size=(1, 50))

X_train = X_train.reshape(1, 50)
t = t.reshape(1, 50)

hidden_3_units = [1, 3, 1]
hidden_20_units = [1, 20, 1]

print('\nTraining with 3 hidden units\n')
nn1 = FullyConnectedRegressor(hidden_3_units, size=50, lr=1e-1)
nn1.fit(X_train, t, epochs=5000)
y_pred1 = nn1.predict(X_train)


print('\nTraining with 20 hidden units\n')
nn2 = FullyConnectedRegressor(hidden_20_units, size=50, lr=1e-1)
nn2.fit(X_train, t, epochs=5000)
y_pred2 = nn2.predict(X_train)


plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.title('3 Hidden Units')
plt.scatter(x=X_train, y=t, label='target')
plt.scatter(x=X_train, y=y_pred1,  label='prediction')
plt.xlabel('epochs')
plt.ylabel('Y_pred & target')
plt.legend()

plt.subplot(222)

plt.title('20 Hidden Units')
plt.scatter(x=X_train, y=t,  label='target')
plt.scatter(x=X_train, y=y_pred2,  label='prediction')
plt.xlabel('epochs')
plt.ylabel('Y_pred & target')
plt.legend()

plt.subplot(223)


plt.plot(nn1.loss_list)
plt.xlabel('epochs')
plt.ylabel('Mean Squared Loss')


plt.subplot(224)

plt.plot(nn2.loss_list)

plt.xlabel('epochs')
plt.ylabel('Mean Squared Loss')

plt.show()


if '__name__' != '__main__':
    print('whats up?')
