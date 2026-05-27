import numpy as np
import pandas as pd

# phenomenological saturation model
Bcmb = 0.0767

# choose saturation scale deep below CMB wavelengths
lambda_star = 1e-4  # meters

# FIRAS frequency range:
# ~30 GHz to ~600 GHz
# convert to wavelength

c = 299792458

nu_min = 30e9
nu_max = 600e9

lam_max = c / nu_min
lam_min = c / nu_max

def B(lam):
    return Bcmb * (1 - np.exp(-lam/lambda_star))

Bmin = B(lam_min)
Bmax = B(lam_max)

deltaB = abs(Bmax - Bmin)

rows = [{
    "lambda_min_m": lam_min,
    "lambda_max_m": lam_max,
    "B_lambda_min": Bmin,
    "B_lambda_max": Bmax,
    "deltaB_FIRAS": deltaB,
    "deltaB_over_Bcmb": deltaB / Bcmb,
}]

df = pd.DataFrame(rows)

print("\nCMB BLACKBODY CONSISTENCY TEST")
print("="*70)
print(df.to_string(index=False))

print("\nInterpretation:")
if deltaB < 1e-5:
    print("FIRAS-compatible: saturation suppresses differential")
    print("chromatic distortion across the CMB band.")
else:
    print("Potential FIRAS inconsistency: differential")
    print("distortion remains too large.")

df.to_csv(
    "results/cmb_blackbody_consistency_test.csv",
    index=False
)

print("\nSaved:")
print("results/cmb_blackbody_consistency_test.csv")
