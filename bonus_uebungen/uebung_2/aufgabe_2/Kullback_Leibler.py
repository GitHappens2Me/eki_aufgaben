import math

dice_rolls = [32, 20, 22, 30, 19, 16, 21, 26]
#dice_rolls = [31, 18, 24, 19, 27, 20, 30, 25]

N = sum(dice_rolls)
print(f"Number of Roles: {N}")

# Wahrscheinlichkeits-verteilung
q = [val / N for val in dice_rolls]
p = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]

print(f"Wahrscheinlichkeits verteilung: {q}")

kl_divergence = 0
for i in range(len(dice_rolls)):
    #print(round(p[i] * math.log( p[i] / q[i]),3))
    kl_divergence += p[i] * math.log( p[i] / q[i])


print(f"Kullback-Leibler-Divergenz(P||Q): {kl_divergence}")


