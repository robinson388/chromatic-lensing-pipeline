import numpy as np
import pandas as pd

np.random.seed(123)

N = 100000

ratios = []
ordering = []

for i in range(N):

    C390 = np.random.uniform(0.01,0.2)

    # strong PSF/drizzle/focus perturbations
    psf_bias = np.random.normal(1.05,0.05)

    C814 = C390 * psf_bias

    R = C814 / C390

    ratios.append(R)
    ordering.append(C814 > C390)

ratios = np.array(ratios)

summary = {
    "P_ordering": np.mean(ordering),
    "R_median": np.median(ratios),
    "R_95": np.percentile(ratios,95),
    "R_99": np.percentile(ratios,99),
    "R_999": np.percentile(ratios,99.9),
    "R_max": np.max(ratios),
}

df = pd.DataFrame([summary])

print("\nAGGRESSIVE PSF NULL TEST")
print("="*70)
print(df.to_string(index=False))

df.to_csv(
    "results/aggressive_psf_null_test.csv",
    index=False
)

print("\nSaved:")
print("results/aggressive_psf_null_test.csv")
