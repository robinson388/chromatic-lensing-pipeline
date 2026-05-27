from pathlib import Path
import pandas as pd

root = Path("/home/robin/lensing_rbp/data/raw/Castles")

rows = []
for f in root.rglob("*.fits"):
    parts = f.parts
    lens = None
    for p in parts:
        if p in ["HE0435","B1608","RXJ0921","RXJ1131","WFI2033","J1004","RXJ0911","PG1115"]:
            lens = p
            break
    if lens is None:
        continue

    banddir = f.parent.parent.name if f.parent.name.startswith("q") else f.parent.name
    # realmente en tu estructura: LENS/FIT/BAND/qxxx.fits
    try:
        band = f.parts[-2]
    except Exception:
        band = "UNKNOWN"

    rows.append({
        "lens": lens,
        "band": band,
        "path": str(f)
    })

df = pd.DataFrame(rows).sort_values(["lens","band","path"])
df.to_csv("results/external_castles_index.csv", index=False)

print(df.to_string(index=False))
