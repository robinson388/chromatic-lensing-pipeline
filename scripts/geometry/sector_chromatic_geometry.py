import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from scipy.ndimage import gaussian_filter

Path("results").mkdir(exist_ok=True)

LENS = "J1630"
NSECTORS = 8

fits_files = list(Path("data").rglob("*.fits")) + list(Path("data").rglob("*.fits.gz"))

def find_file(lens, band):
    for f in fits_files:
        s = str(f).upper()
        if lens.upper() in s and band.upper() in s:
            return f
    return None

def read_fits(path):
    with fits.open(path) as hdul:
        for h in hdul:
            if h.data is not None and np.ndim(h.data) == 2:
                return np.array(h.data, dtype=float)
    raise RuntimeError(f"No 2D data in {path}")

def clean_residual(img):
    img = np.nan_to_num(img, nan=np.nanmedian(img))
    model = gaussian_filter(img, sigma=20)
    resid = img - model
    resid = resid - np.percentile(resid, 5)
    resid = np.clip(resid, 0, None)
    if resid.max() > 0:
        resid = resid / resid.max()
    return gaussian_filter(resid, sigma=2.0)

fb = find_file(LENS, "F390W")
fr = find_file(LENS, "F814W")

if fb is None or fr is None:
    raise FileNotFoundError("No encontré F390W o F814W para " + LENS)

blue = read_fits(fb)
red = read_fits(fr)

# Common central crop: same size, same central region
ny = min(blue.shape[0], red.shape[0])
nx = min(blue.shape[1], red.shape[1])

def center_crop(img, ny, nx):
    y0 = img.shape[0] // 2
    x0 = img.shape[1] // 2
    y1 = y0 - ny // 2
    x1 = x0 - nx // 2
    return img[y1:y1+ny, x1:x1+nx]

blue = center_crop(blue, ny, nx)
red = center_crop(red, ny, nx)

print("Common crop shape:", blue.shape, red.shape)

sb = clean_residual(blue)
sr = clean_residual(red)

# Centro común: máximo de la imagen combinada suavizada
combo = gaussian_filter(blue + red, sigma=5)
y0, x0 = np.unravel_index(np.nanargmax(combo), combo.shape)

yy, xx = np.indices(blue.shape)
r = np.sqrt((xx - x0)**2 + (yy - y0)**2)
phi = np.arctan2(yy - y0, xx - x0)

# Anillo común usando señal combinada residual
scombo = sb + sr
maxr = min(blue.shape) // 2
r_int = r.astype(int)
prof = np.zeros(maxr)

for ri in range(maxr):
    m = r_int == ri
    if m.sum() > 0:
        prof[ri] = np.nanmean(scombo[m])

prof[:8] = 0
prof[int(maxr * 0.85):] = 0
r0 = int(np.argmax(prof))

ann = (r > r0 - 6) & (r < r0 + 6)

vals = scombo[ann]
vals = vals[np.isfinite(vals)]
thr = np.percentile(vals, 80)

common_mask = ann & (scombo >= thr)

edges = np.linspace(-np.pi, np.pi, NSECTORS + 1)
rows = []

for i in range(NSECTORS):
    sec = common_mask & (phi >= edges[i]) & (phi < edges[i+1])

    if sec.sum() < 10:
        continue

    wb = sb[sec] + 1e-12
    wr = sr[sec] + 1e-12

    r390 = np.average(r[sec], weights=wb)
    r814 = np.average(r[sec], weights=wr)

    C390 = abs(np.sum(wb * np.exp(1j * phi[sec])) / np.sum(wb))
    C814 = abs(np.sum(wr * np.exp(1j * phi[sec])) / np.sum(wr))

    rows.append({
        "lens": LENS,
        "sector": i,
        "x0": x0,
        "y0": y0,
        "r0_common": r0,
        "n_pix": int(sec.sum()),
        "r390": r390,
        "r814": r814,
        "delta_r": r814 - r390,
        "C390": C390,
        "C814": C814,
        "delta_C": C814 - C390,
        "phi_min_deg": np.degrees(edges[i]),
        "phi_max_deg": np.degrees(edges[i+1])
    })

df = pd.DataFrame(rows)
out = f"results/{LENS}_sector_chromatic_geometry.csv"
df.to_csv(out, index=False)

print(df)
print("\nSaved:", out)

if len(df) > 0:
    print("\nN sectors:", len(df))
    print("N delta_r < 0:", int((df["delta_r"] < 0).sum()))
    print("median delta_r:", df["delta_r"].median())
    print("N delta_C > 0:", int((df["delta_C"] > 0).sum()))
    print("median delta_C:", df["delta_C"].median())
