import numpy as np
from matplotlib import pyplot as plt

# Einleitung
# #########
# Ähnlich wie in Aufgabe 1 betrachten wir wieder ein Zwei-Klassen Problem aber mit unterschiedlichen
# Datenpunkten
# 
# Rot  (-1,-1), (-1,1), (1,-1), (1,1)
# Blau  (-0.25, 0.0), (0.25, 0.0), (-0.75, 0.0), (0.75, 0.0)

x1 = np.array([ -1, -1,  1, 1, -0.25, 0.25, -0.75, 0.75])
x2 = np.array([ -1,  1, -1, 1,     0,     0,     0,   0])
Y  = np.array([  1,  1,  1, 1,    -1,    -1,    -1,  -1])

plt.scatter(x1[Y == -1], x2[Y == -1], color='blue')
plt.scatter(x1[Y == 1], x2[Y == 1], color='red')


# Aufgabe 2a
# ##########
# Lösen Sie das lineare Regressiondsproblem wie in Aufgabe 1, d.h.
# zeichen Sie zuerst und bestimmen Sie dann die Modellparameter.
#
# Passen Sie dann das Modell an und verwenden Sie stattdessen
#
#   y = w0 + w1 * x1 + w2 * x2
#   y'=  1,       x1,       x2   
# als Model. Schätzen Sie auch hier die Trennfläche und zeichen Sie ebenfalls.

dy_dw0 = np.ones_like(Y)
dy_dw1 = x1
dy_dw2 = x2

# X-Matrix
X = np.stack([dy_dw0, dy_dw1, dy_dw2], axis=1)

W = np.linalg.inv(X.T @ X) @ X.T @ Y

# Change W[2] from 0
W[2] = 0.01

print("X:\n",X)
print("Y:",Y)
print("W:",W)

def f(x1):
    return -(W[0] + W[1] * x1) / (W[2])

lin_x1 = np.linspace(min(x1)- 1, max(x1)+ 1, 100)
lin_x2 = f(lin_x1)
#plt.plot(lin_x1, lin_x2, color='green')


#x1, x2 = np.meshgrid(np.linspace(-2,2,10), np.linspace(-2,2,10))
#z = W[0] + W[1] * x1 + W[2] * x2
#plt.contourf(x1, x2, z, levels=[-10,0,10], colors=["b","r"], alpha=.2)
#plt.contour(x1, x2, z, levels=[0], colors=["k"])
#plt.xlim((-2,2))
#plt.ylim((-2,2))
#plt.show()

# Aufgabe 2b
# ##########
# Schätzen Sie nun wie in der vorherigen Aufgaben ein linears Modell der Form
#
#   y = w0 + w1 * x1 + w2 * x2 + w3 * x2^2
#
# indem Sie die X-Matrix aus der Vorlesung aufstellen, die Pseudoinverse bestimmen und
# dann die Modellparameter bestimmen. 

dy_dw0 = np.ones_like(Y)
dy_dw1 = x1
dy_dw2 = x2
dy_dw3 = x2**2

# X-Matrix
X = np.stack([dy_dw0, dy_dw1, dy_dw2, dy_dw3], axis=1)

W = np.linalg.inv(X.T @ X) @ X.T @ Y


print("X:\n",X)
print("Y:",Y)
print("W:",W)



# Aufgabe 2c
# ##########
# Bringen Sie diesen Code ans Laufen indem Sie ggf. die Variablen w0, w1 und w1 an ihr Skript anpassen.
#
x1, x2 = np.meshgrid(np.linspace(-2,2,10), np.linspace(-2,2,10))
z = W[0] + W[1] * x1 + W[2] * x2  + W[3] * x2 * x2 
plt.contourf(x1, x2, z, levels=[-10,0,10], colors=["b","r"], alpha=.2)
plt.contour(x1, x2, z, levels=[0], colors=["k"])
plt.xlim((-2,2))
plt.ylim((-2,2))
plt.show()