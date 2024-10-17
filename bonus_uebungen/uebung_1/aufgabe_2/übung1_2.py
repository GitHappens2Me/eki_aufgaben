import numpy as np
import matplotlib.pyplot as plt

# Anzahl Kunden / Größe der Messreihe
N_kunden = 6

# Erhobene Daten der Messreihe
Zucker   = np.array([4.7, 3.2, 2.7, 4.5, 4.4, 4.0])
AciBiber = np.array([4.0, 4.1, 3.1, 2.3, 3.2, 4.4])
Salz     = np.array([3.9, 4.5, 3.3, 2.7, 2.4, 4.2])
Lecker   = np.array([  1,   0,   0,   1,    1,  0])   # 1 = Lecker; 0 = Nicht Lecker

# Startparameter (Behauptung des Onkels)
W = np.array([1, 0, 0, -4.0])

# Hilfsfunktion: Sigmoid
def sigmoid(x):
    return ( 1 / (1+np.exp(-x)) )

# Logistisches Modell (Wahrscheinlichkeit, dass Soße lecker ist)
def p_lecker(kunde, W):
    return (
        sigmoid(model(kunde, W))
    )

def p_lecker_2(W, Z, AB, S):
    return (
        sigmoid(model_2(W, Z, AB, S))
    )

# Model: Berechnet die Modellvorhersage. 
def model(kunde, W):
    return W[0] * Zucker[kunde] + W[1] * AciBiber[kunde] + W[2] * Salz[kunde] + W[3]

def model_2(W, Z, AB, S):
    return W[0] * Z + W[1] * AB + W[2] * S + W[3]
    
# Teilaufgabe 1)
# Modellvorhersagen für alle Kunden berechnen:
print("\nTeilaufgabe 1) Modellvorhersagen für alle Kunden berechnen:")
for kunde in range(N_kunden):
    print(f"Kunde {kunde}: Zucker: {Zucker[kunde]}, AciBiber: {AciBiber[kunde]}, Salz: {Salz[kunde]}, Ergebnis: {Lecker[kunde]}, Vorhersage: {round(model(kunde, W), 2)} , Vorhersage der Klasse: {round(p_lecker(kunde, W), 3)}")

# Teilaufgabe 2)
# Likelihood berechnen:
def likelihood(W):
    likelihood = 1
    for kunde in range(N_kunden):
        likelihood *= (np.power(p_lecker(kunde, W), Lecker[kunde]) * np.power(1 - p_lecker(kunde, W), 1 - Lecker[kunde]))
    return likelihood
print("\nTeilaufgabe 2) Likelihood berechnen:")
print(f"Likelihood: {likelihood(W)}")


# Teilaufgabe 3)
# Gradienten bestimmen:

def grad(W):
    return np.array([
        #korrekte indizes bei Zucker, A.b. & Salz? 
        sum([(Lecker[i] - p_lecker(i, W)) * Zucker[i] for i  in range(N_kunden)]),
        sum([(Lecker[i] - p_lecker(i, W)) * AciBiber[i] for i  in range(N_kunden)]),
        sum([(Lecker[i] - p_lecker(i, W)) * Salz[i] for i  in range(N_kunden)]),
        sum([(Lecker[i] - p_lecker(i, W)) * 1 for i  in range(N_kunden)])
    ])
print("\nTeilaufgabe 3) Gradient berechnen:")
print(f"Gradient: {grad(W)}")

# Teilaufgabe 4)
# Gradientenaufstieg (1 Step mit Lernrate 0.1)

def gradient_descend(n, W, lern_rate, log = [], visualisierung = False):
    for i in range(n):

        if(visualisierung and i % 10 == 0):
            plt.clf()
            zucker, aciBiber = np.meshgrid(np.linspace(0, 10, 200), np.linspace(0, 10, 200))
            z =  W[0] * zucker + W[1] * aciBiber + W[3]
            plt.contourf(zucker, aciBiber, z, levels=[-10,0,10], alpha=.2, cmap="coolwarm")
            plt.contour(zucker, aciBiber, z, levels=[0.0], colors=["k"])
            plt.xlabel('Zucker')
            plt.ylabel('AciBiber')
            plt.title("Zusätzliche Visualisierung \n(Verhältnis zwischen Zucker und AciBiber für Leckere Soßen (Ohne Salz))\nVerändert durch Gradient Descend")

            plt.pause(0.05)
           
        log.append(likelihood(W))

        # Anpassen der Parameter
        W = W + lern_rate * grad(W) 
    return W


print("\nTeilaufgabe 4) Gradientenaufstieg (1 Step mit Lernrate 0.1):")
print(f"Parameter nach einem Gradientenaufstieg: {gradient_descend(1, W, lern_rate= 0.01)}")

# Teilaufgabe 5)
# Klassenzugehörigkeitswahrscheinlichkeiten & Likelihood für alternative Parameter
W_alt = [1.145, -0.066, -0.093, -3.991]
print("\nTeilaufgabe 5) Klassenzugehörigkeitswahrscheinlichkeiten & Likelihood für alternative Parameter:")
for kunde in range(N_kunden):
    print(f"Kunde {kunde}: Vorhersage der Klasse: {round(p_lecker(kunde, W_alt), 3)}")
print(f"Likelihood: {likelihood(W_alt)}")

# Teilaufgabe 6)
# Wieviel Zucker ist notwendig bei 3.4g Aci Biber und 3.8g Salz
def benötigter_Zucker(W, Acibiber, Salz):
    return (W[1] * Acibiber + W[2] * Salz + W[3]) / -W[0]
print("\nTeilaufgabe 6) Wieviel Zucker ist notwendig bei 3.4g Aci Biber und 3.8g Salz:")
AB = 3.4
salz = 3.8
print(f"Benötigter Zucker: {benötigter_Zucker(W_alt, AB, 3.8)}")
print(f"Überprüfen mit {round(benötigter_Zucker(W_alt, AB, 3.8),5)} Zucker, {AB} AciBiber & {salz} Salz: {p_lecker_2(W_alt, benötigter_Zucker(W_alt, AB, salz), AB, salz)}")

# Teilaufgabe 7) 
# Methodik hinterfragen

print("\n---------------------Ende der Aufgaben------------------------\n")

# Visualisierung (Verhältnis zwischen Zucker und AciBiber für Leckere Soßen (Ohne Salz) )

steps = 5000
print("Visualisierung (Verhältnis zwischen Zucker und AciBiber für Leckere Soßen (Ohne Salz) )")
print(f"Start-Omega: {W}")
x = np.array([step for step in range(steps)])
y = []
W = gradient_descend(n = steps, W = W, lern_rate= 0.01, log = y, visualisierung=False)
print(f"Finales Omega: {W} (nach {steps} Steps)")
plt.plot(x,y)
plt.show()


# Vorhersage nach Gradient Descend
for kunde in range(N_kunden):
    print(f"Kunde {kunde}: Zucker: {Zucker[kunde]}, AciBiber: {AciBiber[kunde]}, Salz: {Salz[kunde]}, Ergebnis: {Lecker[kunde]}, Vorhersage: {round(model(kunde, W), 2)} , Vorhersage der Klassenzugehörigkeitswahrscheinlichkeit in %: {round(p_lecker(kunde, W), 3)* 100}%")




