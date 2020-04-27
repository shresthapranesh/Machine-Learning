import numpy as np
from feedforward import FullyConnected
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

'''
Please run for multiple times for optimal output, you should get cost function below 5e-2.

'''

X_train = np.array([[-1, 1, -1, 1], [-1, 1, 1, -1]])
y_train = np.array([0, 0, 1, 1]).reshape(1, 4)


layers = [2, 2, 1]

tic = time.time()
nn = FullyConnected(layers, size=4, lr=2e-1)
nn.fit(X_train, y_train, epochs=6000)

toc = time.time()

print('Time elapsed: {} sec'.format(round(toc-tic, 2)))
plt.figure()
plt.plot(nn.loss_list)
plt.xlabel('epochs')
plt.ylabel('log_loss')

xx, yy = np.meshgrid(np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1))
y_ = nn.predict(X_train)
y_pred = nn.predict(np.array([xx.ravel(), yy.ravel()])).reshape(xx.shape)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs=np.array([-1, 1, -1, 1]), ys=np.array([-1, 1, 1, -1]), zs=y_)
ax.plot_surface(X=xx, Y=yy, Z=y_pred)
ax.set_zlabel('Probability')

plt.show()
