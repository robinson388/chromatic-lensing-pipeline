import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from scipy.ndimage import gaussian_filter

def load_fits(path):
    data = fits.getdata(path)
    if data.ndim > 2:
        data = np.squeeze(data)
    data = np.nan_to_num(data.astype(float), nan=0.0)
    return data

def crop_center(img, n=128):
    ny, nx = img.shape
    cy, cx = ny//2, nx//2
    h = n//2
    return img[max(0,cy-h):cy+h, max(0,cx-h):cx+h]

def cdir(img):
    img = crop_center(img, 128)
    img = img - gaussian_filter(img, 8)

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

    return np.abs(np.sum(I*np.exp(1j*th[mask]))) / np.sum(I)

pairs = [
    ("J1004", "AWFCV", "NIC2H",
     "/home/robin/lensing_rbp/data/raw/Castles/J1004/FIT/AWFCV/q128.fits",
     "/home/robin/lensing_rbp/data/raw/Castles/J1004/FIT/NIC2H/q256.fits"),

    ("J1004", "AWFCI", "NIC2H",
     "/home/robin/lensing_rbp/data/raw/Castles/J1004/FIT/AWFCI/q128.fits",
     "/home/robin/lensing_rbp/data/raw/Castles/J1004/FIT/NIC2H/q256.fits"),

    ("RXJ0921", "WFPC2V", "NIC2H",
     "/home/robin/lensing_rbp/data/raw/Castles/RXJ0921/FIT/WFPC2V/q256.fits",
     "/home/robin/lensing_rbp/data/raw/Castles/RXJ0921/FIT/NIC2H/q256.fits"),

    ("RXJ1131", "AWFCV", "NIC2H",
     "/home/robin/lensing_rbp/data/raw/Castles/RXJ1131/FIT/AWFCV/q256.fits",
     "/home/robin/lensing_rbp/data/raw/Castles/RXJ1131/FIT/NIC2H/q256.fits"),

    ("WFI2033", "AWFCV", "NIC2H",
     "/home/robin/lensing_rbp/data/raw/Castles/WFI2033/FIT/AWFCV/q256.fits",
     "/home/robin/lensing_rbp/data/raw/Castles/WFI2033/FIT/NIC2H/q256.fits"),

    ("WFI2033", "AWFCI", "NIC2H",
     "/home/robin/lensing_rbp/data/raw/Castles/WFI2033/FIT/AWFCI/q256.fits",
     "/home/robin/lensing_rbp/data/raw/Castles/WFI2033/FIT/NIC2H/q256.fits"),
]

rows = []
for lens, short_band, long_band, short_path, long_path in pairs:
    short = load_fits(short_path)
    long = load_fits(long_path)

    Cshort = cdir(short)
    Clong = cdir(long)
    ratio = Clong / Cshort if Cshort > 0 else np.nan

    rows.append({
        "lens": lens,
        "short_band": short_band,
        "long_band": long_band,
        "C_short": Cshort,
        "C_long": Clong,
        "R_long_short": ratio,
        "positive": Clong > Cshort,
        "short_path": short_path,
        "long_path": long_path
    })

df = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/external_castles_cdir.csv", index=False)

print(df[["lens","short_band","long_band","C_short","C_long","R_long_short","positive"]].to_string(index=False))
print()
print("Unique lenses positive:")
u = df.groupby("lens")["positive"].max()
print(u.to_string())
print("N positive pairs =", int(df["positive"].sum()), "/", len(df))
print("N positive unique lenses =", int(u.sum()), "/", len(u))
