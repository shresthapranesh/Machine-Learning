import numpy as np
import matplotlib.pyplot as plt

L = 100
N = 25
X = np.random.uniform(0, 1, N)
t = np.sin(2*3.1415*X) + np.random.normal(0, 0.3, N)
