import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, shift
from pathlib import Path

OUT = Path("results")
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(97531)

def cdir(img):
    ny, nx = img.shape
    y, x = np.indices(img.shape)
    cx, cy = nx/2, ny/2
    r = np.hypot(x-cx, y-cy)
    th = np.arctan2(y-cy, x-cx)

    mask = (r > 18) & (r < 55)
    I = img[mask]
    I = I - np.percentile(I, 20)
    I[I < 0] = 0

    if I.sum() <= 0:
        return np.nan

    return np.abs(np.sum(I * np.exp(1j * th[mask]))) / np.sum(I)

def make_nonlens_field(n=128):
    y, x = np.indices((n, n))
    img = np.zeros((n, n))

    # galaxias/campos no-lente: blobs aleatorios sin anillo coherente
    nobj = rng.integers(4, 18)
    for _ in range(nobj):
        x0 = rng.uniform(10, n-10)
        y0 = rng.uniform(10, n-10)
        amp = rng.uniform(0.1, 1.0)
        sx = rng.uniform(1.5, 8.0)
        sy = rng.uniform(1.5, 8.0)
        img += amp*np.exp(-0.5*(((x-x0)/sx)**2 + ((y-y0)/sy)**2))

    # fondo suave
    gx, gy = rng.normal(0, 0.3, 2)
    img += 0.05 + gx*(x-n/2)/n + gy*(y-n/2)/n
    img = np.clip(img, 0, None)
    return img

def drizzle_like_noise(shape, sigma=0.02, corr=1.0):
    return gaussian_filter(rng.normal(0, sigma, shape), corr)

N = 10000
rows = []

for i in range(N):
    base = make_nonlens_field()

    img390 = base.copy()
    img814 = base.copy()

    # PSF filtro
    img390 = gaussian_filter(img390, rng.uniform(1.0, 1.9))
    img814 = gaussian_filter(img814, rng.uniform(1.8, 3.3))

    # registro
    img390 = shift(img390, rng.normal(0, 0.18, 2), order=1, mode="nearest")
    img814 = shift(img814, rng.normal(0, 0.18, 2), order=1, mode="nearest")

    # ruido correlacionado
    img390 += drizzle_like_noise(img390.shape, rng.uniform(0.01, 0.05), rng.uniform(0.5, 1.8))
    img814 += drizzle_like_noise(img814.shape, rng.uniform(0.01, 0.05), rng.uniform(0.5, 1.8))

    C390 = cdir(img390)
    C814 = cdir(img814)
    R = C814 / C390 if C390 > 0 else np.nan

    rows.append([i, C390, C814, R])

df = pd.DataFrame(rows, columns=["trial","C390","C814","ratio"])
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df.to_csv(OUT / "forward_null_v3_nonlens_controls.csv", index=False)

print(df["ratio"].describe(percentiles=[0.95,0.99,0.999]))
print("P(C814>C390) =", (df["C814"] > df["C390"]).mean())
print("max ratio =", df["ratio"].max())
print("N valid =", len(df))
