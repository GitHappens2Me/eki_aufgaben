import numpy as np
from matplotlib import pyplot as plt

# Einleitung
# #########
# Gegeben sind nun diese Daten
coords_x1 = np.random.normal(0, 0.5, 300) 
coords_x2 = np.random.normal(0, 0.5, 300)
class_y = 2.0 * ((coords_x1**2 + coords_x2**2) < 0.4) - 1.0


plt.scatter(coords_x1[class_y == -1], coords_x2[class_y == -1], color='blue')
plt.scatter(coords_x1[class_y ==  1], coords_x2[class_y == 1], color='red')


# Aufgabe 3a
# ##########
# Schätzen Sie ein Modell der Form 
#
#   y = w0 +      w1 * x1 +       w2 * x2 +       w3 * x1^2 +      w4 * x2^2 +      w5*x1*x2
# 
# und zeichen Sie die Daten sowie die Trennfläche ihres Modells


dy_dw0 = np.ones_like(class_y)
dy_dw1 = coords_x1
dy_dw2 = coords_x2
dy_dw3 = coords_x1**2
dy_dw4 = coords_x2**2
dy_dw5 = coords_x1 * coords_x2


# X-Matrix
X = np.stack([dy_dw0, dy_dw1, dy_dw2, dy_dw3, dy_dw4, dy_dw5], axis=1)

W = np.linalg.inv(X.T @ X) @ X.T @ class_y



print("X:\n",X)
print("Y:",class_y)
print("W:",W)


## Bringen Sie diesen Code wieder ans Laufen um die Trennfläche zu visualieren
x1, x2 = np.meshgrid(np.linspace(-2,2,50), np.linspace(-2,2,50))
z = W[0] + W[1] * x1 + W[2] * x2 + W[3] * (x1 ** 2) + W[4] * (x2 ** 2) + W[5] * x1 * x2
z = np.clip(z, -1, 1)
print(np.min(z), np.max(z))
blue_indices = (class_y < 0.0)
red_indices = (class_y > 0.0)
plt.contourf(x1, x2, z, levels=np.linspace(-1,1,20), cmap="coolwarm", alpha=.2)
plt.contour(x1, x2, z, levels=[0.0], colors=["k"])
plt.plot(coords_x1[blue_indices], coords_x2[blue_indices], 'bo')
plt.plot(coords_x1[red_indices], coords_x2[red_indices], 'ro')
plt.xlim((-2,2))
plt.ylim((-2,2))

plt.show()