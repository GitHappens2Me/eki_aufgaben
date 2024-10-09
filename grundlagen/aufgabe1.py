# Activate venv mit: 
# .\myvenv\Scripts\activate
# 

import numpy as np
from matplotlib import pyplot as plt


print(np.__version__)

# Aufgabe 1
############

# Erzeugen Sie mit np.linspace ein NumPy Array mit 100 Werten zwischen 0 und 15
# Berechnen Sie dann die dazugehörigen Funktionswerte der Funktion
#   
#   f(x) = exp(sin(x))
#
# Leiten Sie f ab und berechnen Sie ebenfalls die Funktionswerte der Ableitung
#
# Plotten Sie f(x) und f'(x) mit plt.plot(...) und plt.show(...)


def f(x): return np.exp(np.sin(values))
def df_dx(x): return np.exp(np.sin(values)) * np.cos(values)

values = np.linspace(0, 15, 100)
print(values)
#plt.plot(values)

f_values = f(values)
print(f_values)
plt.plot(f_values, label='f')

dfdx_values = df_dx(values)
print(dfdx_values)
plt.plot(dfdx_values, label="f'")





## Aufgabe 1b
#############

# Approximieren Sie die Ableitung nummerisch über
# 
#   f'(x) ~ dy/dx = (f(x+1) - f(x-1)) / dx
# 
# Überlegen Sie was genau dx in diesem Zusammenhang ist und plotten Sie 
# ihre numerische Approximation ins gleiche Koordinatensystem

# Approximate derivative numerically


x = values
y = f(values)
                                            # y        [ 1, 2, 3, 4, 5, 6,]
n = (y[1:] - y[:-1]) / (x[1:] - x[:-1])    #  y[1:]:   [ 2, 3, 4, 5, 6, ]
                                           #  y[:-1]   [ 1, 2, 3, 4, 5, ]

test = [1 , 2 , 3,4, 5,6]

print(len(x[1:]))
print(len(n))

plt.plot( n, label="f' (Aprox)")


plt.axhline()

plt.legend()
plt.show()
