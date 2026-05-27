import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.stats import binomtest, wilcoxon
import matplotlib.pyplot as plt

print("Iniciando cosmic prism residual-mask geometry...")

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

fits_files = list(Path("data").rglob("*.fits")) + list(Path("data").rglob("*.fits.gz"))

lenses = [
    "J1627","J0737","J0912","J1630","J0252",
    "J0903","J0956","J0959","J1430","J2341"
]

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

def residual_mask_geometry(path, sig=2.0):
    img = read_fits(path)
    img = np.nan_to_num(img, nan=np.nanmedian(img))

    model = gaussian_filter(img, sigma=20)
    resid = img - model
    resid = resid - np.percentile(resid, 5)
    resid = np.clip(resid, 0, None)

    if resid.max() > 0:
        resid = resid / resid.max()

    sm = gaussian_filter(resid, sigma=sig)

    y0, x0 = np.unravel_index(np.argmax(gaussian_filter(img, 5)), img.shape)

    yy, xx = np.indices(img.shape)
    r = np.sqrt((xx - x0)**2 + (yy - y0)**2)
    phi = np.arctan2(yy - y0, xx - x0)

    maxr = min(img.shape) // 2
    r_int = r.astype(int)
    prof = np.zeros(maxr)

    for ri in range(maxr):
        m = r_int == ri
        if m.sum() > 0:
            prof[ri] = np.nanmean(sm[m])

    prof[:8] = 0
    prof[int(maxr * 0.85):] = 0
    r0 = int(np.argmax(prof))

    ann = (r > r0 - 6) & (r < r0 + 6)

    vals = sm[ann]
    vals = vals[np.isfinite(vals)]
    if len(vals) < 20:
        return None

    thr = np.percentile(vals, 80)
    mask = ann & (sm >= thr)

    if mask.sum() < 10:
        return None

    w = sm[mask] + 1e-12

    r_mean = np.average(r[mask], weights=w)
    sinm = np.average(np.sin(phi[mask]), weights=w)
    cosm = np.average(np.cos(phi[mask]), weights=w)
    phi_mean = np.arctan2(sinm, cosm)

    Cdir = abs(np.sum(w * np.exp(1j * phi[mask])) / np.sum(w))

    return {
        "r0": r0,
        "r_mean": r_mean,
        "phi_mean": phi_mean,
        "Cdir_recomputed": Cdir,
        "x0": x0,
        "y0": y0,
        "n_mask": int(mask.sum())
    }

rows = []

for lens in lenses:
    print("Procesando", lens)

    fb = find_file(lens, "F390W")
    fr = find_file(lens, "F814W")

    if fb is None or fr is None:
        rows.append({
            "lens_id": lens,
            "status": "missing"
        })
        continue

    gb = residual_mask_geometry(fb)
    gr = residual_mask_geometry(fr)

    if gb is None or gr is None:
        rows.append({
            "lens_id": lens,
            "status": "bad"
        })
        continue

    dphi = np.arctan2(
        np.sin(gr["phi_mean"] - gb["phi_mean"]),
        np.cos(gr["phi_mean"] - gb["phi_mean"])
    )

    rows.append({
        "lens_id": lens,
        "status": "ok",
        "blue_file": str(fb),
        "red_file": str(fr),
        "r390": gb["r_mean"],
        "r814": gr["r_mean"],
        "delta_r": gr["r_mean"] - gb["r_mean"],
        "phi390": gb["phi_mean"],
        "phi814": gr["phi_mean"],
        "delta_phi_rad": dphi,
        "delta_phi_deg": np.degrees(dphi),
        "C390_recomputed": gb["Cdir_recomputed"],
        "C814_recomputed": gr["Cdir_recomputed"],
        "delta_C_recomputed": gr["Cdir_recomputed"] - gb["Cdir_recomputed"],
        "n390": gb["n_mask"],
        "n814": gr["n_mask"]
    })

df = pd.DataFrame(rows)
df.to_csv("results/cosmic_prism_residual_mask_geometry.csv", index=False)

print("\n=== RESIDUAL-MASK COSMIC PRISM GEOMETRY ===")
print(df)

ok = df[df["status"] == "ok"].copy()

if len(ok) >= 3:
    n = len(ok)

    n_dC_pos = int((ok["delta_C_recomputed"] > 0).sum())
    n_dr_pos = int((ok["delta_r"] > 0).sum())
    n_dp_pos = int((ok["delta_phi_rad"] > 0).sum())

    summary = pd.DataFrame({
        "quantity": [
            "N_ok",
            "N_delta_C_positive",
            "sign_p_delta_C",
            "median_delta_C",
            "N_delta_r_positive",
            "sign_p_delta_r",
            "median_delta_r_pix",
            "mean_delta_r_pix",
            "wilcoxon_p_delta_r",
            "N_delta_phi_positive",
            "sign_p_delta_phi",
            "median_delta_phi_deg",
            "mean_delta_phi_deg",
            "wilcoxon_p_delta_phi"
        ],
        "value": [
            n,
            n_dC_pos,
            binomtest(n_dC_pos, n, 0.5, alternative="greater").pvalue,
            ok["delta_C_recomputed"].median(),
            n_dr_pos,
            binomtest(n_dr_pos, n, 0.5).pvalue,
            ok["delta_r"].median(),
            ok["delta_r"].mean(),
            wilcoxon(ok["delta_r"]).pvalue,
            n_dp_pos,
            binomtest(n_dp_pos, n, 0.5).pvalue,
            ok["delta_phi_deg"].median(),
            ok["delta_phi_deg"].mean(),
            wilcoxon(ok["delta_phi_rad"]).pvalue
        ]
    })

    summary.to_csv("results/cosmic_prism_residual_mask_geometry_summary.csv", index=False)

    print("\n=== SUMMARY ===")
    print(summary)

    plt.figure(figsize=(7,5))
    plt.bar(ok["lens_id"], ok["delta_r"])
    plt.axhline(0, linestyle="--")
    plt.ylabel("delta r = r814 - r390 [pix]")
    plt.title("Residual-mask cosmic prism: radial shift")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figures/cosmic_prism_residual_mask_delta_r.png", dpi=200)

    plt.figure(figsize=(7,5))
    plt.bar(ok["lens_id"], ok["delta_phi_deg"])
    plt.axhline(0, linestyle="--")
    plt.ylabel("delta phi [deg]")
    plt.title("Residual-mask cosmic prism: angular shift")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figures/cosmic_prism_residual_mask_delta_phi.png", dpi=200)

    print("\nSaved:")
    print("results/cosmic_prism_residual_mask_geometry.csv")
    print("results/cosmic_prism_residual_mask_geometry_summary.csv")
    print("figures/cosmic_prism_residual_mask_delta_r.png")
    print("figures/cosmic_prism_residual_mask_delta_phi.png")
else:
    print("\nNo hay suficientes lentes válidos.")
