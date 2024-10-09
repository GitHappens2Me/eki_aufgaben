import numpy as np
from matplotlib import pyplot as plt

# Einleitung
############
# Wir haben wieder eine Meßreihe, diesmal mit 100 Datenpunkten.
noise = 0.1
coords_x = np.linspace(0,6.28,100)
coords_y = 1.0 + 2.0 * np.sin(coords_x) + 3.0 * np.cos(coords_x)
coords_x = coords_x + np.random.normal(0.0, noise, coords_x.shape)
coords_y = coords_y + np.random.normal(0.0, noise, coords_y.shape)

# Aufgabe 2a
############
# Plotten Sie die Koordinaten in ein Koordinatensystem um sich einen 
# Überblick zu verschaffen

plt.plot(coords_x, coords_y, label="Data", linestyle="", marker="o", markersize=2)


# Aufgabe 2b
############
# Schätzen Sie nun wie in Aufgabe 1 ein lineares Modell der Form
#
#   y = w0 + w1 * sin(x) + w2 * cos(x)
#
# Geben Sie die geschätzten Modellparameter auf der Konsole aus

def fun(x, w):
    return w[0] + w[1]*np.sin(x) + w[2]*np.cos(x)

# Create THE X-Matrix
ones_column = np.ones_like(coords_x)       # Spalte mit Einsen für w0
column1 = np.sin(coords_x)                     # Spalte für w[1]* np.sin(x)
column2 = np.cos(coords_x)            # Spalte für w[2]*np.cos(x)


# X-Matrix durch Stacken der Spalten
X = np.stack((ones_column, column1, column2), axis=1)
print(X)

# Pseudo-Inverse Berechnen
inv = np.linalg.inv(X.T @ X) @ X.T

# Omega berechnen
w = inv @ coords_y
print(w)



# Aufgabe 2c
############
# Plotten Sie die durch ihre Modellparameter geschätzte Funktion in das gleiche Koordinatensystem

lin_x = np.linspace(0 ,10, 100)
plt.plot(lin_x, fun(lin_x, w), label="Koordinaten")
 


plt.legend()
plt.show()