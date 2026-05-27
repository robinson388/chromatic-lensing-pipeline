import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, shift
from pathlib import Path

OUT = Path("results")
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(24680)

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

def make_clumpy_arc(n=128):
    y, x = np.indices((n, n))
    cx, cy = n/2, n/2
    r = np.hypot(x-cx, y-cy)
    th = np.arctan2(y-cy, x-cx)

    ring_radius = rng.uniform(34, 42)
    ring_width = rng.uniform(2.5, 5.5)
    ring = np.exp(-0.5*((r-ring_radius)/ring_width)**2)

    # Fuente extendida asimétrica
    morph = (
        1.0
        + rng.uniform(0.2, 0.7)*np.cos(th-rng.uniform(-np.pi, np.pi))
        + rng.uniform(0.1, 0.5)*np.cos(2*th-rng.uniform(-np.pi, np.pi))
        + rng.uniform(0.05, 0.35)*np.sin(3*th-rng.uniform(-np.pi, np.pi))
    )
    morph = np.clip(morph, 0, None)

    arc = ring * morph

    # Clumps tipo galaxia fuente
    nclumps = rng.integers(3, 10)
    for _ in range(nclumps):
        ang = rng.uniform(-np.pi, np.pi)
        rr = rng.normal(ring_radius, ring_width)
        x0 = cx + rr*np.cos(ang)
        y0 = cy + rr*np.sin(ang)
        amp = rng.uniform(0.15, 1.2)
        sig = rng.uniform(1.0, 4.0)
        arc += amp*np.exp(-0.5*(((x-x0)**2 + (y-y0)**2)/sig**2))

    # Núcleo/lente suave residual
    lens_core = rng.uniform(0.15, 0.45)*np.exp(-0.5*(r/rng.uniform(7,13))**2)

    return arc + lens_core

def apply_color_gradient(img, strength):
    # Gradiente cromático artificial permitido como morfología de fuente,
    # no como lente cromática.
    ny, nx = img.shape
    y, x = np.indices(img.shape)
    gx, gy = rng.normal(0, 1, 2)
    grad = gx*(x-nx/2)/nx + gy*(y-ny/2)/ny
    grad = 1 + strength*grad
    return img * np.clip(grad, 0.4, 1.8)

def drizzle_like_noise(shape, sigma=0.015, corr=1.0):
    noise = rng.normal(0, sigma, shape)
    return gaussian_filter(noise, corr)

N = 10000
rows = []

for i in range(N):
    # Misma física/lente achromática
    base = make_clumpy_arc()

    # Diferencias de morfología de fuente por color, pero moderadas
    img390_true = apply_color_gradient(base.copy(), rng.uniform(0.0, 0.6))
    img814_true = apply_color_gradient(base.copy(), rng.uniform(0.0, 0.6))

    # PSF dependiente del filtro
    psf390 = rng.uniform(1.0, 1.9)
    psf814 = rng.uniform(1.8, 3.3)

    img390 = gaussian_filter(img390_true, psf390)
    img814 = gaussian_filter(img814_true, psf814)

    # Registro subpixel
    dx390, dy390 = rng.normal(0, 0.18, 2)
    dx814, dy814 = rng.normal(0, 0.18, 2)

    img390 = shift(img390, (dy390, dx390), order=1, mode="nearest")
    img814 = shift(img814, (dy814, dx814), order=1, mode="nearest")

    # Drizzle / ruido correlacionado
    img390 += drizzle_like_noise(img390.shape, sigma=rng.uniform(0.01, 0.05), corr=rng.uniform(0.5, 1.8))
    img814 += drizzle_like_noise(img814.shape, sigma=rng.uniform(0.01, 0.05), corr=rng.uniform(0.5, 1.8))

    C390 = cdir(img390)
    C814 = cdir(img814)
    R = C814 / C390 if C390 > 0 else np.nan

    rows.append([i, C390, C814, R, psf390, psf814])

df = pd.DataFrame(rows, columns=["trial","C390","C814","ratio","psf390","psf814"])
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df.to_csv(OUT / "forward_null_v2_clumpy_source.csv", index=False)

print(df["ratio"].describe(percentiles=[0.95,0.99,0.999]))
print("P(C814>C390) =", (df["C814"] > df["C390"]).mean())
print("max ratio =", df["ratio"].max())
print("N valid =", len(df))
