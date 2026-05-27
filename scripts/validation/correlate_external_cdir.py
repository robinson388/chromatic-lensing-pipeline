import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

df = pd.read_csv("results/external_castles_cdir.csv")

# Metadatos aproximados conocidos para estos sistemas.
# Si luego tenemos valores mejores, los reemplazamos.
meta = pd.DataFrame([
    {"lens":"J1004",   "z_lens":0.680, "z_source":1.734},
    {"lens":"RXJ0921", "z_lens":0.320, "z_source":1.660},
    {"lens":"RXJ1131", "z_lens":0.295, "z_source":0.658},
    {"lens":"WFI2033", "z_lens":0.657, "z_source":1.662},
])

df = df.merge(meta, on="lens", how="left")
df["logR"] = np.log10(df["R_long_short"])
df["z_ratio"] = df["z_source"] / df["z_lens"]
df["dz"] = df["z_source"] - df["z_lens"]

rows = []
for x in ["z_lens", "z_source", "z_ratio", "dz"]:
    sub = df.dropna(subset=[x, "logR"])
    if len(sub) >= 3:
        sp = spearmanr(sub[x], sub["logR"])
        pr = pearsonr(sub[x], sub["logR"])
        rows.append({
            "x": x,
            "N": len(sub),
            "spearman_rho": sp.statistic,
            "spearman_p": sp.pvalue,
            "pearson_r": pr.statistic,
            "pearson_p": pr.pvalue,
        })

corr = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/external_castles_cdir_with_metadata.csv", index=False)
corr.to_csv("results/external_castles_correlations.csv", index=False)

print("DATA:")
print(df[["lens","short_band","long_band","R_long_short","logR","z_lens","z_source","z_ratio","dz","positive"]].to_string(index=False))
print()
print("CORRELATIONS:")
print(corr.to_string(index=False))
