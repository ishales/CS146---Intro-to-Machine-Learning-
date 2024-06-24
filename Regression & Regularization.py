
#HW 1 M146 Isabelle Hales

import os
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 4,0],[1, 1,1],[1, 0,1],[1, -2,-2],[1,-2,1],[1,1,0],[1,5,2],[1,3,0]])
y = np.array([12, 3, 1, 6, 3, 6, 8, 7])
theta = np.array([1.0, 1.0, 1.0])

n = len(X)


xA = np.dot(X, theta)
loss = xA - y


G_jmse = (1/n)*(np.dot(np.transpose(X), loss))
print(G_jmse)

G_jmae = (1/n)*(np.dot(np.transpose(X),(np.sign(loss))))
print(G_jmae)


