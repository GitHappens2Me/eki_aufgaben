import numpy as np
import matplotlib.pyplot as plt

N_kunden = 6

Zucker = np.array([4.7, 3.2, 2.7, 4.5, 4.4, 4.0])
AciBiber = np.array([4.0, 4.1, 3.1, 2.3, 3.2, 4.4])
Salz = np.array([3.9, 4.5, 3.3, 2.7, 2.4, 4.2])
Lecker = np.array([1, 0, 0, 1, 1, 0])

W = np.array([1, 0, 0, -4.0])

def sigmoid(x):
    return ( 1 / (1+np.exp(-x)) )

def p_lecker(kunde, W):
    return (
        sigmoid(W[0] * Zucker[kunde] + W[1] * AciBiber[kunde] + W[2] * Salz[kunde] + W[3])
    )



# Vorherhsagen berechnen:
for kunde in range(N_kunden):
    print(f"Kunde {kunde}: Zucker: {Zucker[kunde]}, AciBiber: {AciBiber[kunde]}, Salz: {Salz[kunde]}, Ergebnis: {Lecker[kunde]}, Vorhersage: {p_lecker(kunde, W)}")



# Likelihood berechnen:

def likelihood():
    likelihood = 1
    for kunde in range(N_kunden):
        likelihood *= (np.power(p_lecker(kunde, W), Lecker[kunde]) * np.power(1 - p_lecker(kunde, W), 1 - Lecker[kunde]))
    return likelihood

print(f"Likelihood: {likelihood()}")


# Gradienten bestimmen:

def grad(W):
    return np.array([
        #korrekte indizes bei Zucker, A.b. & Salz? 
        (Lecker[0] - p_lecker(0, W) * Zucker[0]),
        (Lecker[1] - p_lecker(1, W) * AciBiber[1]),
        (Lecker[2] - p_lecker(2, W) * Salz[2]),
        (Lecker[3] - p_lecker(3, W) * 1 )
    ])

print(f"Gradient: {grad(W)}")

# Gradienten Abstieg:

def gradient_descend(n, W, lern_rate):
    for i in range(n):
        W = W + lern_rate * grad(W) 
    return W

# Für Übung: n = 1
# Frage: Warum bleibt Kunde 0 schlecht vorhergesagt?
print(f"Omega: {W}")
W = gradient_descend(n = 100000, W = W, lern_rate= 0.1)
print(f"Omega: {W}")
for kunde in range(N_kunden):
    print(f"Kunde {kunde}: Zucker: {Zucker[kunde]}, AciBiber: {AciBiber[kunde]}, Salz: {Salz[kunde]}, Ergebnis: {Lecker[kunde]}, Vorhersage: {p_lecker(kunde, W)}")


