import numpy as np

x = np.array([ 1.6, 1, -1.4, 0, -0.4])
Y = np.array([ -1.8, -0.8, -1.6, 1.2, 0])

dy_dw0 = np.ones_like(Y)
dy_dw1 = x
dy_dw2 = x*x
dy_dw3 = x*x*x

# X-Matrix
X = np.stack([dy_dw0, dy_dw1, dy_dw2, dy_dw3], axis=1)

print("X:", X)
print("Y:", Y)


W = np.linalg.inv(X.T @ X) @X.T @ Y
print("W:", W)