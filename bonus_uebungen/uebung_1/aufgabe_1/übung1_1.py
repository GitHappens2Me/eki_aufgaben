import numpy as np
import matplotlib.pyplot as plt


x1 = np.array([-4, -3, -2, -1, 0, 2, 4, 5, 7, 8])
x2 = np.array([ 5,  6,  3,  2, 1,-1,-4,-5,-2,-3])
Y  = np.array([ 0,  0,  0,  0, 0, 1, 1, 1, 1, 1])

# X-Matrix
X = np.array([  [1, -4, 16, 5],
                [1, -3, 9, 6],
                [1, -2, 4, 3],
                [1, -1, 1, 2],
                [1, 0, 0, 1],
                [1, 2, 4, -1],
                [1, 4, 16, -4],
                [1, 5, 25, -5],
                [1, 7, 49, -2],
                [1, 8, 64, -3]])

# Omega
W = np.linalg.inv(X.T @ X) @X.T @ Y

print("X:\n",X)
print("Y:",Y)
print("W:",W)


# Plot Points
plt.scatter(x1[Y== 0], x2[Y == 0], color='blue')
plt.scatter(x1[Y == 1], x2[Y == 1], color='red')

# Plot the Parabel
lin_x1 = np.linspace(min(x1)- 1, max(x1)+ 1, 100)
x2_boundary = -(W[0] + W[1] * lin_x1 + W[2] * (lin_x1 ** 2)) / W[3]  
plt.plot(lin_x1, x2_boundary, color='green')


plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Lineare Klassifikation')
plt.legend()

plt.grid(True)
plt.show()