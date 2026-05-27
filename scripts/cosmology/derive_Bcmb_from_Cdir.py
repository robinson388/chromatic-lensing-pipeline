import numpy as np

# Datos principales del paper
lenses = {
    "J1627": {"C390": 0.0280, "C814": 0.6470},
    "J0737": {"C390": 0.0360, "C814": 0.2310},
    "J0912": {"C390": 0.0075, "C814": 0.0499},
    "J1630": {"C390": 0.0720, "C814": 0.8940},
    "J0252": {"C390": 0.0466, "C814": 0.0951},
}

# Modelo mínimo:
# Bcmb_pred = eta * mean(C814 - C390)
# Buscamos eta físico mínimo para que el contraste cromático observado
# produzca el sesgo CMB requerido.

Bcmb_required = 0.0767

deltaCs = []
ratios = []

for name, d in lenses.items():
    deltaC = d["C814"] - d["C390"]
    ratio = d["C814"] / d["C390"]
    deltaCs.append(deltaC)
    ratios.append(ratio)

mean_deltaC = np.mean(deltaCs)
median_deltaC = np.median(deltaCs)
mean_ratio = np.mean(ratios)

eta_mean = Bcmb_required / mean_deltaC
eta_median = Bcmb_required / median_deltaC

print("\n=== DERIVE Bcmb FROM Cdir TEST ===\n")
print("Required Bcmb:", Bcmb_required)
print("Mean DeltaC:", mean_deltaC)
print("Median DeltaC:", median_deltaC)
print("Mean C814/C390 ratio:", mean_ratio)

print("\nRequired eta using mean DeltaC:", eta_mean)
print("Required eta using median DeltaC:", eta_median)

print("\nPredicted Bcmb per lens using eta_mean:")
for name, d in lenses.items():
    deltaC = d["C814"] - d["C390"]
    Bpred = eta_mean * deltaC
    print(name, "DeltaC =", deltaC, "Bcmb_pred =", Bpred)

print("\nInterpretation:")
print("If eta is order 0.1-1, the amplitude is not extreme.")
print("If eta is huge, the model is probably tuned.")
