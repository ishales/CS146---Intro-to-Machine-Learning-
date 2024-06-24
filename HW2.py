
#HW 1 M146 Isabelle Hales

import os
import numpy as np
import matplotlib.pyplot as plt

# Question 4a
A_x1 = np.array([4,-2,1])
A_x2 = np.array([0,1,0])
plt.scatter(A_x1,A_x2, c='blue')

B_x1 = np.array([1,3,5])
B_x2 = np.array([1,-1,2,])
plt.scatter(B_x1,B_x2, c='red')

plt.ylim(-5, 5)
plt.xlim(-5, 5)
plt.show()

