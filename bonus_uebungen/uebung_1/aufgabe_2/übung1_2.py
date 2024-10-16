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
        sigmoid(model(kunde, W))
    )

def model(kunde, W):
    return W[0] * Zucker[kunde] + W[1] * AciBiber[kunde] + W[2] * Salz[kunde] + W[3]
    

# Vorherhsagen berechnen:
for kunde in range(N_kunden):
    print(f"Kunde {kunde}: Zucker: {Zucker[kunde]}, AciBiber: {AciBiber[kunde]}, Salz: {Salz[kunde]}, Ergebnis: {Lecker[kunde]}, Vorhersage: {round(model(kunde, W), 2)} , Vorhersage%: {round(p_lecker(kunde, W), 3)* 100}%")


# Likelihood berechnen:

def likelihood(W):
    likelihood = 1
    for kunde in range(N_kunden):
        likelihood *= (np.power(p_lecker(kunde, W), Lecker[kunde]) * np.power(1 - p_lecker(kunde, W), 1 - Lecker[kunde]))
    return likelihood

print(f"Likelihood: {likelihood(W)}")



# Gradienten bestimmen:

def grad(W):
    return np.array([
        #korrekte indizes bei Zucker, A.b. & Salz? 
        sum([(Lecker[i] - p_lecker(i, W)) * Zucker[i] for i  in range(N_kunden)]),
        sum([(Lecker[i] - p_lecker(i, W)) * AciBiber[1] for i  in range(N_kunden)]),
        sum([(Lecker[i] - p_lecker(i, W)) * Salz[2] for i  in range(N_kunden)]),
        sum([(Lecker[i] - p_lecker(i, W)) * 1 for i  in range(N_kunden)])
    ])

# [(Lecker[i] - p_lecker(i, W) * W[0]) for i  in range(N_kunden)],
print(f"Gradient: {grad(W)}")

# Gradienten Abstieg:

def gradient_descend(n, W, lern_rate, log):
   
    
    for i in range(n):
        #print(f"{i}: {likelihood(W)}, {grad(W)}")
        
        if(i % 10 == 0):
            plt.clf()

            zucker, aciBiber = np.meshgrid(np.linspace(0, 10, 200), np.linspace(0, 10, 200))
            z =  W[0] * zucker + W[1] * aciBiber + W[3]
            plt.contourf(zucker, aciBiber, z, levels=[-10,0,10], alpha=.2, cmap="coolwarm")
            plt.contour(zucker, aciBiber, z, levels=[0.0], colors=["k"])
            plt.xlabel('Zucker')
            plt.ylabel('AciBiber')

            plt.pause(0.1)

        log.append(likelihood(W))
        W = W + lern_rate * grad(W) 
    return W


steps = 5000

# Für Übung: n = 1
# Frage: Warum bleibt Kunde 0 schlecht vorhergesagt?
print(f"Omega: {W}")

x = np.array([step for step in range(steps)])
y = []



W = gradient_descend(n = steps, W = W, lern_rate= 0.01, log = y)
print(f"Omega: {W}")

#plt.plot(x, y)
#plt.grid()




for kunde in range(N_kunden):
    print(f"Kunde {kunde}: Zucker: {Zucker[kunde]}, AciBiber: {AciBiber[kunde]}, Salz: {Salz[kunde]}, Ergebnis: {Lecker[kunde]}, Vorhersage: {round(model(kunde, W), 2)} , Vorhersage%: {round(p_lecker(kunde, W), 3)* 100}%")


plt.show()



