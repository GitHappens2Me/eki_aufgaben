import numpy as np
from matplotlib import pyplot as plt


## Aufgabe 2a
#############
#
# Wir wollen im folgenden die Funktion
#
#   f(x,y,z) = (x-3)^2 + (y-2)^2 + (z+5)^2
#
# betrachten. Bestimmen Sie den Gradieten der Funktion und implementieren Sie zwei
# Funktionen f und grad_f. Die Funktion f soll zu einem Tripel (x,y,z) den dazugehörigen
# Funktionswert an der gegebenen Stelle liefern. Die Funktion grad_f soll zu einem Tripel (x,y,z)
# den Gradienten liefern. 
#
# Hinweis: Sie können ihre Funktionen mit bekannten Werten testen, so ist z.B.
#
#   f(3,2,-5) = grad_f(3,2,-5) = (0,0,0)
#   
# und
#
#   f(0,0,0) = 38 während grad_f(0,0,0) = (-6,-4,10)
#
# ist.


#  f(x,y,z) = (x-3)^2 + (y-2)^2 + (z+5)^2
def f(x,y,z):
    return (x-3)**2 + (y-2)**2 + (z+5)**2


def f_grad(x,y,z):
    return  np.array( 
            [2*(x-3),
            2*(y-2), 
            2*(z+5)])

print(f(3,2,-5), f_grad(3,2,-5))
print(f(0,0,0), f_grad(0,0,0))

## Aufgabe 2b:
##############
#
# Implementieren Sie nun ausgehend vom Startpunkt x_0 = (12, -5, 9)
# ein Gradientenabstiegsverfahren mit einer Lernrate von eta=0.05
# Berechen Sie dazu iterativ 100 mal sowohl der Funktionswert als auch
# den Gradienten am aktuellen Punkt. Setzen Sie dann
#
#   x_(n+1) = x_n - eta * grad_f(x_n)) 
#
# Geben Sie in jedem Schritt sowohl die Koordinate x_n aus sowie den 
# Gradienten an der Stelle.

Step_n = 100

x_0 = np.array([12, -5, 9])
x_n = np.array([12, -5, 9])

x_array = []
y_array = []
z_array = []

# Start Gradient Descend
eta = 0.03
for i in range(Step_n):
    grad = f_grad(x_n[0], x_n[1], x_n[2])
    x_n = x_n - eta * grad
    print("Step",i,": Koordinaten:", x_n, " Gradient:", grad)
    x_array.append(x_n[0])
    y_array.append(x_n[1])
    z_array.append(x_n[2])
    

# Show Convergence to optimal Values

lin_x = np.linspace(0 ,Step_n, Step_n)
plt.plot(lin_x, x_array, label="X")
plt.plot(lin_x, y_array, label="Y")
plt.plot(lin_x, z_array, label="Z")
 


plt.legend()
plt.show()