
import numpy as np
import matplotlib.pyplot as plt

# exercise 1
mean = [0,0]
cov = np.eye(2)

dataA = np.random.multivariate_normal(mean,cov,1000)
x1a = dataA[:, 0]
x2a = dataA[:, 1]

plt.scatter(x1a, x2a)
plt.title("Question 1")
plt.show()

# exercise 2
mean2 = [-1,1]
dataB = np.random.multivariate_normal(mean2,cov,1000)
x1b = dataB[:, 0]
x2b = dataB[:, 1]

plt.scatter(x1b, x2b)
plt.title("Question 2")
plt.show()

# exercise 3
cov2 = np.eye(2) * 2
dataC = np.random.multivariate_normal(mean,cov2,1000)
x1c = dataC[:, 0]
x2c = dataC[:, 1]

plt.scatter(x1c, x2c)
plt.title("Question 3")
plt.show()

# exercise 4
cov3 = [[1, 0.5],
        [0.5, 1]]
dataD = np.random.multivariate_normal(mean,cov3,1000)
x1d = dataD[:, 0]
x2d = dataD[:, 1]

plt.scatter(x1d, x2d)
plt.title("Question 4")
plt.show()

# exercise 5
cov4 = [[1, -0.5],
        [-0.5, 1]]
dataE = np.random.multivariate_normal(mean,cov4,1000)
x1e = dataE[:, 0]
x2e = dataE[:, 1]

plt.scatter(x1e, x2e)
plt.title("Question 5")
plt.show()
