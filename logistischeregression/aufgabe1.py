import numpy as np
from matplotlib import pyplot as plt
from aufgabe1_misc import zeichnen

# Einleitung
# #########
# Gegeben sind nun diese Daten

coords_x1 = np.random.normal(0, 0.5, 100) 
coords_x2 = np.random.normal(0, 0.5, 100)
dx, dy = np.random.uniform(-0.4, 0.4, 2)
class_y = (((coords_x1-dx)**2 + (coords_x2-dy)**2) < 0.4)
sample_size = len(coords_x1)
# Ausserdem wurde bereits mit Hilfe der linearen Regression ein Modell der Form
#
#   y = w0 + w1*x1 + w2*x2 + w3*x1^2 + w4*x2^2 + w5*x1*x2 (Gl. 1)
#
# geschätzt.
X = np.stack([np.ones_like(coords_x1), coords_x1, coords_x2, coords_x1**2, coords_x2**2, coords_x1 * coords_x2], axis=1)
Xinv = np.linalg.inv(X.T @ X) @ X.T
W = Xinv @ (2.0*class_y-1.0)

W += np.random.uniform(-0.5, 0.5, W.shape)

# Der folgende Code zeichnet das Model (vergleiche Aufgabe 3 bei der linearen Regression)
print(W)
zeichnen(W, coords_x1, coords_x2, class_y)
plt.pause(2)


# Aufgabe 1
###########
# Sie sollen das gefundene Modell nun mit Hilfe der logistischen Regression iterativ verbessern.
# Berechnen Sie dazu zunächst die Klassenzugehörigkeitswahrscheinlichkeiten mit Hilfe der Sigmoid-Funktion.
#
#   P(Y = 1 | x) = 1 / (1 + exp(-y))    (Gl.2)
#
# Hinweis: Das y in (Gl. 2) ist die von der linearen Regression vorhergesagte Variable (vgl. Gl. 1). Berechnen Sie
# diese zuerst.
#
# Hinweis 2: So können sie überprüfen ob ihre Wahrscheinlichkeiten korrekt berechnen werden:
#   Angenommen Sie haben die Wahrscheinlichkeiten in einem Array P berechet. 
#   Berechnen Sie zuerst mit (P > 0.5) diejenigen Punkte, die das Modell der positiven Klasse zuordnen würde
#   Berechnen Sie dann mit (P > 0.5) == class_y, welche Punkte ihrer korrekten Klasse zugeordnet werden.
#   Geben Sie mit print((P>0.5) == class_y) dieses Array auf der Konsole aus. Die meisten Einträge sollten "True" sein.

def model(W, x1, x2):
    return  W[0] + W[1]*x1 + W[2]*x2 + W[3]*x1*x1 + W[4]*x2*x2 + W[5]*x1*x2 

def sigmoid(x):
    return ( 1 / (1+np.exp(-x)) )

def p(W, x1, x2):
    return (
        sigmoid(model(W, x1, x2))
    )

def p_i(W, i):
    return (
        sigmoid(model(W, coords_x1[i], coords_x2[i]))
    )

print()
print(coords_x1)
class_preditions = [p(W, coords_x2[index], coords_x2[index]) for index in range(len(coords_x1))]
for i, pred in enumerate(class_preditions):
    print(f"Index: {i}: Prediction: {round(pred,2)}, Correct: {class_y[i]}")


# Aufgabe 2
###########
# Berechen Sie die Likelihood ihres Models wie in der Vorlesung gezeigt und geben 
# Sie diese auf der Konsole aus.
#
# Hinweis: Es ist 
#
#   L = Produkt p_i^y_i * (1.0-p_i)^(1.0 - y_i) (Gl. 4)
#
# Sie können np.power und np.prod verwenden um die Potenzen bzw. das Produkt zu berechnen

def likelihood(W):
    likelihood = 1
    for i in range(len(coords_x1)):
        likelihood *= (np.power(p(W, coords_x1[i], coords_x2[i]), class_y[i]) * np.power(1 - p(W, coords_x1[i], coords_x2[i]), 1 - class_y[i]))
    return likelihood
 

def likelihood_mean(W):
    lkh = likelihood(W)
    return np.power(lkh, 1 / sample_size) 





print(likelihood(W))

# Aufgabe 3
###########
# Berechnen Sie nun den Gradienten der logitischen Regression wie in der Vorlesung gezeigt.
# Beachten Sie das gilt
#
#   dL/dwi = Summe (y_i - p_i) * dy/dwi     (Gl. 3)
#
# Dabei ist y_i die wahre Klasse (in diesem Skript class_y), p_i die vorhergesagte Wahrscheinlichkeit aus Aufgabe 1
# und dy/dwi entspricht der partiellen Ableitung des Modells (vgl. Gl. 1) nach den Parametern. 
#
# Sie können np.sum verwenden um die Summe zu berechnen


# W[0] + W[1]*x1 + W[2]*x2 + W[3]*x1*x1 + W[4]*x2*x2 + W[5]*x1*x2 
def grad(W):
    x1 = coords_x1
    x2 = coords_x2
    return np.array([
        
        sum([(class_y[i] - p_i(W,i)) * 1 for i  in range(sample_size)]),
        sum([(class_y[i] - p_i(W,i)) * x1[i] for i  in range(sample_size)]),
        sum([(class_y[i] - p_i(W,i)) * x2[i] for i  in range(sample_size)]),
        sum([(class_y[i] - p_i(W,i)) * x1[i] * x1[i] for i  in range(sample_size)]),
        sum([(class_y[i] - p_i(W,i)) * x2[i] * x2[i] for i  in range(sample_size)]),
        sum([(class_y[i] - p_i(W,i)) * x1[i] * x2[i] for i  in range(sample_size)])
    ])

print(grad(W))


# Aufgabe 4
###########
# Führen Sie 1000 Schritte eines Gradientenaufstiegsverfahren (Aufstieg!) durch mit 
# einer Lernrate von eta = 0.005. Geben Sie in jedem Schritt die Likelihood ihres Modells aus (vgl. oben)
# Zeichnen Sie in jedem Schritt das neue Modell mit Hilfe der Zeichnen Funktion (vgl. Einleitung) und 
# erzeugen Sie eine kurze Pause mit plt.pause(0.05)
#
# Hinweis: Beachten Sie das Sie nach jedem Gradientenupdate natürlich die Modelvorhersage (Gl. 1),
# die Klassenzugehörigkeitswahrscheinlichkeit (Gl. 2) und den Gradienten (Gl. 3) neu berechnen müssen. 


likelihood_log = []
lernrate = 0.01

def gradient_descend(n, W, lern_rate, log = [], visualisierung = True):
    for i in range(n):

        if(visualisierung):# and (i % 10 == 0)):
            plt.clf()
            x1, x2 = np.meshgrid(np.linspace(-1, 1, 500), np.linspace(-1, 1, 500))
            z =  model(W, x1, x2)
            plt.contourf(x1, x2, z, levels=[-1,0,1], alpha=.2, cmap="coolwarm")
            plt.contour(x1, x2, z, levels=[0.0], colors=["k"])
            zeichnen(W, coords_x1, coords_x2, class_y)
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.title("Gradient Ascend")
            print(f"Step {i}: Likelihood: {likelihood_mean(W)}")
            plt.pause(0.05)
        
        log.append(likelihood(W))

        # Anpassen der Parameter
        W = W + lern_rate * grad(W) 
        
    return W

gradient_descend(1000, W,lernrate,likelihood_log)
plt.pause(3)
#print(likelihood_log)



for iter in range(1000):
    pass

    # Berechnen Sie hier die neuen Klassenzugehörigkeitswahrscheinlichkeiten (Gl. 2)
    
    # Geben Sie hier die Likelihood ihres aktuellen Modells aus (Gl. 4)
    
    # Berechnen Sie hier den Gradienten (Gl. 3)

    # Führen Sie hier das eigentlichen Gradientenupdate aus und zeichnen Sie neu
    #       w1 = w0 + 0.05 * grad
