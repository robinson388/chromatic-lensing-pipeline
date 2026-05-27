import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.stats import binomtest, wilcoxon
import matplotlib.pyplot as plt

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

# Busca FITS automáticamente
fits_files = list(Path("data").rglob("*.fits")) + list(Path("data").rglob("*.fits.gz"))

blue_keys = ["F390W", "390"]
red_keys  = ["F814W", "814"]

def read_fits(path):
    with fits.open(path) as hdul:
        for h in hdul:
            if h.data is not None and np.ndim(h.data) == 2:
                return np.array(h.data, dtype=float)
    return None

def find_file(lens, keys):
    cand = []
    for f in fits_files:
        name = f.name.upper()
        full = str(f).upper()
        if lens.upper() in full and any(k.upper() in name or k.upper() in full for k in keys):
            cand.append(f)
    return cand[0] if cand else None

def geom_stats(img):
    img = np.nan_to_num(img, nan=0.0)
    img = img - np.nanmedian(img)
    img[img < 0] = 0

    sm = gaussian_filter(img, 2.0)
    y0, x0 = np.unravel_index(np.argmax(sm), sm.shape)

    yy, xx = np.indices(img.shape)
    r = np.sqrt((xx-x0)**2 + (yy-y0)**2)
    phi = np.arctan2(yy-y0, xx-x0)

    # máscara robusta: top 5% de señal positiva, excluyendo núcleo
    vals = sm[sm > 0]
    if len(vals) < 20:
        return None

    thr = np.percentile(vals, 95)
    mask = (sm >= thr) & (r > 3)

    if mask.sum() < 10:
        return None

    w = sm[mask]
    r_mean = np.average(r[mask], weights=w)

    # media circular ponderada
    sinm = np.average(np.sin(phi[mask]), weights=w)
    cosm = np.average(np.cos(phi[mask]), weights=w)
    phi_mean = np.arctan2(sinm, cosm)

    return r_mean, phi_mean, x0, y0, mask.sum()

lenses = ["J1627","J0737","J0912","J1630","J0252",
          "J0903","J0956","J0959","J1430","J2341"]

rows = []

for lens in lenses:
    fb = find_file(lens, blue_keys)
    fr = find_file(lens, red_keys)

    if fb is None or fr is None:
        rows.append({
            "lens_id": lens,
            "blue_file": str(fb) if fb else "MISSING",
            "red_file": str(fr) if fr else "MISSING",
            "status": "missing_files"
        })
        continue

    ib = read_fits(fb)
    ir = read_fits(fr)

    gb = geom_stats(ib)
    gr = geom_stats(ir)

    if gb is None or gr is None:
        rows.append({
            "lens_id": lens,
            "blue_file": str(fb),
            "red_file": str(fr),
            "status": "bad_geometry"
        })
        continue

    rb, phib, xb, yb, nb = gb
    rr, phir, xr, yr, nr = gr

    dphi = np.arctan2(np.sin(phir-phib), np.cos(phir-phib))

    rows.append({
        "lens_id": lens,
        "blue_file": str(fb),
        "red_file": str(fr),
        "status": "ok",
        "r390": rb,
        "r814": rr,
        "delta_r": rr-rb,
        "phi390": phib,
        "phi814": phir,
        "delta_phi_rad": dphi,
        "delta_phi_deg": np.degrees(dphi),
        "x390": xb,
        "y390": yb,
        "x814": xr,
        "y814": yr,
        "n_pix390": nb,
        "n_pix814": nr
    })

df = pd.DataFrame(rows)
df.to_csv("results/cosmic_prism_geometry_results.csv", index=False)

ok = df[df["status"]=="ok"].copy()

print("\n=== COSMIC PRISM GEOMETRY RESULTS ===")
print(df)

if len(ok) >= 3:
    k_r_pos = int((ok["delta_r"] > 0).sum())
    n = len(ok)
    p_r = binomtest(k_r_pos, n, 0.5, alternative="two-sided").pvalue

    k_phi_pos = int((ok["delta_phi_rad"] > 0).sum())
    p_phi = binomtest(k_phi_pos, n, 0.5, alternative="two-sided").pvalue

    try:
        w_r = wilcoxon(ok["delta_r"]).pvalue
    except Exception:
        w_r = np.nan

    try:
        w_phi = wilcoxon(ok["delta_phi_rad"]).pvalue
    except Exception:
        w_phi = np.nan

    summary = pd.DataFrame({
        "quantity": [
            "N_ok",
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
            k_r_pos,
            p_r,
            ok["delta_r"].median(),
            ok["delta_r"].mean(),
            w_r,
            k_phi_pos,
            p_phi,
            ok["delta_phi_deg"].median(),
            ok["delta_phi_deg"].mean(),
            w_phi
        ]
    })

    summary.to_csv("results/cosmic_prism_geometry_summary.csv", index=False)

    print("\n=== GEOMETRY SUMMARY ===")
    print(summary)

    plt.figure(figsize=(7,5))
    plt.bar(ok["lens_id"], ok["delta_r"])
    plt.axhline(0, linestyle="--")
    plt.ylabel(r"$\Delta r = r_{814}-r_{390}$ [pix]")
    plt.title("Cosmic Prism Geometry Test: radial shift")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figures/cosmic_prism_delta_r.png", dpi=200)

    plt.figure(figsize=(7,5))
    plt.bar(ok["lens_id"], ok["delta_phi_deg"])
    plt.axhline(0, linestyle="--")
    plt.ylabel(r"$\Delta \phi = \phi_{814}-\phi_{390}$ [deg]")
    plt.title("Cosmic Prism Geometry Test: angular shift")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figures/cosmic_prism_delta_phi.png", dpi=200)

    print("\nSaved:")
    print("results/cosmic_prism_geometry_results.csv")
    print("results/cosmic_prism_geometry_summary.csv")
    print("figures/cosmic_prism_delta_r.png")
    print("figures/cosmic_prism_delta_phi.png")

else:
    print("\nNo hay suficientes lentes con FITS azul y rojo encontrados automáticamente.")
    print("Revisa los nombres con:")
    print("find data -iname '*390*' -o -iname '*814*'")
