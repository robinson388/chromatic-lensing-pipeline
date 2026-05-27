import numpy as np
import pandas as pd

# Datos del paper: C390 y C814
data = [
    ("J1627", 0.0280, 0.6470),
    ("J0737", 0.0360, 0.2310),
    ("J0912", 0.0075, 0.0499),
    ("J1630", 0.0720, 0.8940),
    ("J0252", 0.0466, 0.0951),
]

df = pd.DataFrame(data, columns=["lens_id", "C390", "C814"])

# contraste cromático observado
df["DeltaC"] = df["C814"] - df["C390"]
df["ratio"] = df["C814"] / df["C390"]

# Modelo fenomenológico mínimo:
# delta_z / z = k * normalized(DeltaC)
# Queremos saber qué k haría falta para producir 5% de sesgo en H0.
target_bias = 0.05

mean_deltaC = df["DeltaC"].mean()
df["DeltaC_norm"] = df["DeltaC"] / mean_deltaC

# k necesario para que el promedio produzca 5%
k_required = target_bias / df["DeltaC_norm"].mean()

df["delta_z_over_z_required"] = k_required * df["DeltaC_norm"]
df["DeltaH0_over_H0"] = -df["delta_z_over_z_required"]

print("\n=== CHROMATIC → H0 BIAS BRIDGE TEST ===\n")
print(df)
print("\nMean DeltaC:", mean_deltaC)
print("Required coupling k for 5% H0 bias:", k_required)
print("Mean |DeltaH0/H0|:", np.mean(np.abs(df["DeltaH0_over_H0"])))

df.to_csv("results/cosmology/Cdir_to_H0_bias_bridge.csv", index=False)

print("\nSaved:")
print("results/cosmology/Cdir_to_H0_bias_bridge.csv")
