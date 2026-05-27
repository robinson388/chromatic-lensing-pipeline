import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path

lambda_map = {
    "F390W": 390,
    "F435W": 435,
    "F555W": 555,
    "F606W": 606,
    "F814W": 814,
    "F160W": 1600,
}

def compute_cdir(image):
    image = np.nan_to_num(image.astype(float), nan=0.0)

    ny, nx = image.shape
    y, x = np.indices((ny, nx))
    cx, cy = nx // 2, ny // 2

    x = x - cx
    y = y - cy

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    mask = (r > 10) & (r < min(nx, ny) * 0.40)

    I = image[mask]
    theta = theta[mask]

    I = I - np.nanmedian(I)
    I[I < 0] = 0

    if np.sum(I) <= 0:
        return np.nan

    return np.abs(np.sum(I * np.exp(1j * theta))) / np.sum(I)

rows = []

for path in Path("data").glob("*multiband*/*.fits"):
    fname = path.name.upper()
    folder = path.parent.name

    filt = None
    for key in lambda_map:
        if key in fname:
            filt = key
            break

    if filt is None:
        continue

    try:
        with fits.open(path) as hdul:
            img = None
            for hdu in hdul:
                if hdu.data is not None:
                    img = hdu.data
                    break
        if img is None:
            continue

        if img.ndim > 2:
            img = img[0]

        C = compute_cdir(img)

        lens = folder.replace("_multiband_alt", "").replace("_multiband", "")

        rows.append({
            "lens": lens.upper(),
            "filter": filt,
            "lambda_nm": lambda_map[filt],
            "Cdir": C,
            "file": str(path)
        })

    except Exception as e:
        print("SKIP:", path, e)
        continue

if len(rows) == 0:
    print("No valid FITS read.")
    raise SystemExit

df = pd.DataFrame(rows)
df = df.sort_values(["lens", "lambda_nm"])

df.to_csv("results/chromatic_multiband_auto.csv", index=False)

print("Saved: results/chromatic_multiband_auto.csv")
print(df)
print("\nN rows =", len(df))
print("N lenses =", df["lens"].nunique() if len(df) else 0)
