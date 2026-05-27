import numpy as np

# modelo saturado
alpha=2.12
Bmax=0.0767
lambda0=5e-4

lam_bao = 650e-9   # observaciones ópticas típicas
lam_cmb = 2e-3

def bias(lam):
    return Bmax*(1-np.exp(-(lam/lambda0)**alpha))

b_bao=bias(lam_bao)
b_cmb=bias(lam_cmb)

# shift relativo entre probes
relative_shift=b_cmb-b_bao

print("\n=== BAO CONSISTENCY TEST ===")
print("BAO bias:",b_bao)
print("CMB bias:",b_cmb)
print("Relative CMB-BAO shift:",relative_shift)

if relative_shift < 0.02:
    print("\nCompatible-ish with precision BAO.")
elif relative_shift <0.05:
    print("\nTension with BAO likely.")
else:
    print("\nProbably too large for BAO.")
