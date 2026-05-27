import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path
from scipy.ndimage import gaussian_filter

lambda_map = {
    "F390W": 390,
    "F435W": 435,
    "F555W": 555,
    "F606W": 606,
    "F814W": 814,
    "F160W": 1600,
}

def read_fits(path):
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                img = hdu.data
                if img.ndim > 2:
                    img = img[0]
                return np.nan_to_num(img.astype(float), nan=0.0)
    return None

def crop_center(img, size=160):
    ny, nx = img.shape
    cx, cy = nx // 2, ny // 2
    half = size // 2
    return img[max(0,cy-half):min(ny,cy+half), max(0,cx-half):min(nx,cx+half)]

def compute_cdir_refined(img):
    img = crop_center(img, size=min(180, min(img.shape)))

    # quitar fondo suave/luz de galaxia
    smooth = gaussian_filter(img, sigma=8)
    resid = img - smooth

    # usar solo residuales positivos: arcos/estructura
    resid[resid < 0] = 0

    ny, nx = resid.shape
    y, x = np.indices((ny, nx))
    cx, cy = nx // 2, ny // 2

    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    # máscara anular refinada
    rmin = 8
    rmax = min(nx, ny) * 0.42
    mask = (r > rmin) & (r < rmax)

    I = resid[mask]
    th = theta[mask]

    # quedarnos con píxeles significativos
    if len(I) == 0 or np.nanmax(I) <= 0:
        return np.nan

    threshold = np.percentile(I[I > 0], 70) if np.any(I > 0) else 0
    sel = I > threshold

    I = I[sel]
    th = th[sel]

    if len(I) < 20 or np.sum(I) <= 0:
        return np.nan

    C = np.abs(np.sum(I * np.exp(1j * th))) / np.sum(I)
    return C

rows = []

for path in Path("data").rglob("*.fits"):
    if "multiband" not in str(path).lower():
        continue

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
        img = read_fits(path)
        if img is None:
            continue

        C = compute_cdir_refined(img)

        lens = folder.replace("_multiband_alt", "").replace("_multiband", "")
        lens = lens.upper()

        rows.append({
            "lens": lens,
            "filter": filt,
            "lambda_nm": lambda_map[filt],
            "Cdir": C,
            "file": str(path)
        })

    except Exception as e:
        print("SKIP:", path, e)
        continue

if len(rows) == 0:
    print("No valid refined measurements.")
    raise SystemExit

df = pd.DataFrame(rows)
df = df.dropna(subset=["Cdir"])
df = df.sort_values(["lens", "lambda_nm"])

df.to_csv("results/chromatic_multiband_refined.csv", index=False)

print("Saved: results/chromatic_multiband_refined.csv")
print(df)
print("\nN rows =", len(df))
print("N lenses =", df["lens"].nunique())
