import numpy as np
from matplotlib import pyplot as plt

# Einleitung
# #########
# In einem Zwei-Klassenproblem habe wir folgende Punkte gemessen
# 
# Rot  (1,3), (2,2), (3,3), (3,2), (2,4)
# Blau (3,5), (5,3), (5,4), (6,5), (4,4)

# Aufgabe 1a
# ##########
# Definieren Sie mit NumPy ein coords_x1 und coords_x2 Array für die gegebenen Punkte
# Definieren Sie weiterhin ein class_y Array für die Klassen der gegebenen Punkte (-1 und 1)
# Verwenden Sie dann plt.plot um die roten und blauen Punkte in ihrer jeweiligen Farbe zu zeichnen. 

x1 = np.array([ 1,  2,  3,  3, 2, 3, 5, 5, 6, 4])
x2 = np.array([ 3,  2,  3,  2, 4, 5, 3, 4, 5, 4])
Y  = np.array([ 1,  1,  1,  1, 1, -1, -1, -1, -1, -1])

# Plot Points
plt.scatter(x1[Y == -1], x2[Y == -1], color='blue')
plt.scatter(x1[Y == 1], x2[Y == 1], color='red')

# Aufgabe 1b
# ##########
# Schätzen Sie nun wie in der vorherigen Aufgaben ein linears Modell der Form
#
#   y = w0 + w1 * x1 + w2 * x2 
#   (Gl. 1)
#
# indem Sie die X-Matrix aus der Vorlesung aufstellen, die Pseudoinverse bestimmen und
# dann die Modellparameter bestimmen. 
dy_dw0 = np.ones_like(Y)
dy_dw1 = x1
dy_dw2 = x2
# X-Matrix
X = np.stack([dy_dw0, dy_dw1, dy_dw2], axis=1)

W = np.linalg.inv(X.T @ X) @X.T @ Y

print("X:\n",X)
print("Y:",Y)
print("W:",W)


# Aufgabe 1c
# ##########
# Mit dem Modell aus Aufgabe 1c (Gl. 1) ist die Grenzfläche zwischen beiden Klassen 
# durch die Gleichung
#
#   y = w0 + w1 * x1 + w2 * x2 = 0 (Gl. 2)
#
# definiert. Lösen Sie Gl. 2 nach x2 auf und zeichen Sie diese Trennebene ebenfalls ein

def f(x1):
    return -(W[0] + W[1] * x1) / (W[2])

lin_x1 = np.linspace(min(x1)- 1, max(x1)+ 1, 100)
lin_x2 = f(lin_x1)
#plt.plot(lin_x1, lin_x2, color='green')



# Aufgabe 1d
# ##########
# Bringen Sie diesen Code ans Laufen indem Sie ggf. die Variablen w0, w1 und w1 an ihr Skript anpassen.
#
x1, x2 = np.meshgrid(np.linspace(-11,11,10), np.linspace(-11,11,10))
z = W[0] + W[1] * x1 + W[2] * x2 
plt.contourf(x1, x2, z, levels=[-10,0,10], colors=["b","r"], alpha=.2)
plt.contour(x1, x2, z, levels=[0], colors=["k"])
plt.xlim((0,7))
plt.ylim((0,6))
plt.show()


