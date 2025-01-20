import numpy as np
import matplotlib.pyplot as plt

# Datensatz: 
x1 = np.array([-4, -3, -2, -1, 0, 2, 4, 5, 7, 8])
x2 = np.array([ 5,  6,  3,  2, 1,-1,-4,-5,-2,-3])
Y  = np.array([-1, -1, -1, -1,-1, 1, 1, 1, 1, 1])


# X-Matrix aufstellen 
df_dw0 = np.ones_like(Y)
df_dw1 = x1
df_dw2 = x1**2
df_dw3 = x2
X = np.stack([df_dw0, df_dw1, df_dw2, df_dw3], axis=1)

# Omega berechnen:
W = np.linalg.inv(X.T @ X) @ X.T @ Y

# Ausgabe der Resultate:
print(f"Die X-Matrix hat folgene Form:\n{X}\n")
print(f"Die Pseudoinverse ist: \n{(np.linalg.inv(X.T @ X) @ X.T).round(2)}\n")
print(f"Die berechneten Paramter werte sind:{W}\n")



# Visualisierung der Trennebene und der Datenpunkte
x1_mesh, x2_mesh = np.meshgrid(np.linspace(-8,10,50), np.linspace(-8,10,50))
z = W[0] + W[1] * x1_mesh + W[2] * x1_mesh * x1_mesh + W[3] * x2_mesh
z = np.clip(z, -1, 1)

plt.contourf(x1_mesh, x2_mesh, z, levels=np.linspace(-1,1,20), cmap="coolwarm", alpha=.2)
plt.contour(x1_mesh, x2_mesh, z, levels=[0.0], colors=["k"])
plt.plot(x1[(Y < 0.0)], x2[(Y < 0.0)], 'bs')  # Blaue Vierecke
plt.plot(x1[(Y > 0.0)], x2[(Y > 0.0)], 'r^')  # Rote Dreiecke
plt.xlim((-5,10))
plt.ylim((-8,10))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Lineare Klassifikation')
plt.grid(True)
plt.show()