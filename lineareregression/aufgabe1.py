import numpy as np
from matplotlib import pyplot as plt

# Einleitung
############
# Wir haben eine Meßreihe mit 4 Datenpunkten aufgenommen und gemessen
#
#      x | 1 | 2 | 3 | 4
#       -----------------
#      y | 3 | 4 | 4 | 2
#
# Wir möchten ein linears Modell der Form
#
#   y = w0 + w1*x + w2*x² + w3x³ 
#
# schätzen und dazu die vier Parameter w0, w1, w2 und w3 bestimmen.

# Aufgabe 1a
############
# Definieren Sie geeignte NumPy Arrays um die gegebenen Daten verwenden zu können.
# Verwenden Sie die Funktionen np.ones_like und np.stack um die aus der Vorlesung bekannte
# X-Matrix für das gegebenen Modell aufzustellen.

# Gemessene Werte

# x_data = np.array([1, 2, 3, 4])
# y_data = np.array([3, 4, 4, 2])

x_data = np.array([1, 2, 3, 4, 5, 6])
y_data = np.array([3, 4, 4, 2, 2, 1])

# geschätzte Funktion abhängig von x & w
def fun(x, w):
    return w[0] + w[1]*x + w[2]*x*x + w[3]*x*x*x 

# Create THE X-Matrix
ones_column = np.ones_like(x_data)       # Spalte mit Einsen für w0
x_column = x_data                        # Spalte für w1 * x
x_squared_column = x_data**2             # Spalte für w2 * x^2
x_cubed_column = x_data**3               # Spalte für w3 * x^3

# X-Matrix durch Stacken der Spalten
X = np.stack((ones_column, x_column, x_squared_column, x_cubed_column), axis=1)

print(X)


# Aufgabe 1b
############
# Bestimmen Sie nun mittel np.linalg.inv die Pseudeoinverse von X sowie die Modellparameter w0,w1,w2 und w3
# Geben Sie diese auf der Konsole aus. Berechnen Sie ebenfalls
# den summarischen quadratischen Fehler ihrer Vorhersage und geben Sie auch diese auf der Konsole aus.

# Pseudo-Inverse Berechnen
inv = np.linalg.inv(X.T @ X) @ X.T

# Omega berechnen
w = inv @ y_data
print(w)




# Aufgabe 1c
############
# Plotten Sie die Meßreihe sowie das geschätzte Modell 

lin_x = np.linspace(0,10, 100)
plt.plot(lin_x, fun(lin_x,w), label="Pred.")

plt.plot(x_data, fun(x_data,w), label="Data", linestyle="",marker="o")



# Aufgabe 1d
############
# Was verändert sich wenn Sie ihrer Meßreihe weitere Koordinaten hinzufügen, z.B den
# Punkt (5,2)?

plt.legend()
plt.show()