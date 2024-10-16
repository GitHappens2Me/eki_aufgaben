import numpy as np
import matplotlib.pyplot as plt


coords_x1 = np.array([-4, -3, -2, -1, 0, 2, 4, 5, 7, 8])
coords_x2 = np.array([ 5,  6,  3,  2, 1,-1,-4,-5,-2,-3])
Y  = np.array       ([-1, -1, -1, -1,-1, 1, 1, 1, 1, 1])

# Plot Points
plt.scatter(coords_x1[Y== -1], coords_x2[Y == -1], color='blue')
plt.scatter(coords_x1[Y == 1], coords_x2[Y == 1], color='red')


# X-Matrix
df_dw0 = np.ones_like(Y)
df_dw1 = coords_x1
df_dw2 = coords_x1**2
df_dw3 = coords_x2

X = np.stack([df_dw0, df_dw1, df_dw2, df_dw3], axis=1)

# Omega
W = np.linalg.inv(X.T @ X) @ X.T @ Y

print("X:\n",X)
print("Y:",Y)
print("W:",W)


# Plot the Parabel
#lin_x1 = np.linspace(min(x1)- 1, max(x1)+ 1, 100)
#x2_boundary = -(W[0] + W[1] * lin_x1 + W[2] * (lin_x1 ** 2)) / W[3]  
#plt.plot(lin_x1, x2_boundary, color='green')

x1, x2 = np.meshgrid(np.linspace(-5,10,50), np.linspace(-5,10,50))
z = W[0] + W[1] * x1 + W[2] * x1 * x1 + W[3] * x2
z = np.clip(z, -1, 1)
print(np.min(z), np.max(z))
blue_indices = (Y < 0.0)
red_indices = (Y > 0.0)
plt.contourf(x1, x2, z, levels=np.linspace(-1,1,20), cmap="coolwarm", alpha=.2)
plt.contour(x1, x2, z, levels=[0.0], colors=["k"])
plt.plot(coords_x1[blue_indices], coords_x2[blue_indices], 'bo')
plt.plot(coords_x1[red_indices], coords_x2[red_indices], 'ro')
plt.xlim((-5,10))
plt.ylim((-5,10))


plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Lineare Klassifikation')
plt.legend()

plt.grid(True)
plt.show()