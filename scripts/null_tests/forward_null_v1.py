import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, shift
from pathlib import Path

OUT = Path("results")
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(12345)

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

def make_arc(n=128):
    y, x = np.indices((n, n))
    cx, cy = n/2, n/2
    r = np.hypot(x-cx, y-cy)
    th = np.arctan2(y-cy, x-cx)

    ring = np.exp(-0.5*((r-38)/4.0)**2)
    source_morph = (
        1.0
        + 0.45*np.cos(th-0.7)
        + 0.25*np.cos(2*th+1.3)
        + 0.15*np.sin(3*th)
    )
    arc = ring * np.clip(source_morph, 0, None)

    lens_core = 0.35*np.exp(-0.5*(r/10)**2)
    return arc + lens_core

def drizzle_like_noise(shape, sigma=0.015, corr=1.0):
    noise = rng.normal(0, sigma, shape)
    return gaussian_filter(noise, corr)

N = 10000
rows = []

base = make_arc()

for i in range(N):
    # intrinsic achromatic truth
    intrinsic_390 = base.copy()
    intrinsic_814 = base.copy()

    # filter-dependent PSF
    psf390 = rng.uniform(1.0, 1.7)
    psf814 = rng.uniform(1.8, 3.0)

    img390 = gaussian_filter(intrinsic_390, psf390)
    img814 = gaussian_filter(intrinsic_814, psf814)

    # subpixel registration
    dx390, dy390 = rng.normal(0, 0.12, 2)
    dx814, dy814 = rng.normal(0, 0.12, 2)

    img390 = shift(img390, (dy390, dx390), order=1, mode="nearest")
    img814 = shift(img814, (dy814, dx814), order=1, mode="nearest")

    # drizzle/correlated noise
    img390 += drizzle_like_noise(img390.shape, sigma=rng.uniform(0.01, 0.04), corr=rng.uniform(0.6, 1.4))
    img814 += drizzle_like_noise(img814.shape, sigma=rng.uniform(0.01, 0.04), corr=rng.uniform(0.6, 1.4))

    C390 = cdir(img390)
    C814 = cdir(img814)
    R = C814 / C390 if C390 > 0 else np.nan

    rows.append([i, C390, C814, R, psf390, psf814])

df = pd.DataFrame(rows, columns=["trial","C390","C814","ratio","psf390","psf814"])
df.to_csv(OUT / "forward_null_v1.csv", index=False)

print(df["ratio"].describe(percentiles=[0.95,0.99,0.999]))
print("P(C814>C390) =", (df["C814"] > df["C390"]).mean())
print("max ratio =", df["ratio"].max())
