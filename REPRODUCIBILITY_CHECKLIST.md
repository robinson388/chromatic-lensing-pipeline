# Reproducibility Checklist

## PASS: Achromatic null tests

### forward_null_v1.py
Status: PASS  
Interpretation: narrow achromatic GR-like null. Ratios remain close to unity.

Expected output:
- mean ratio ~1.025
- 99.9% ratio ~1.08
- max ratio ~1.08
- P(C814>C390) ~0.93

### forward_null_v2_clumpy_source.py
Status: PASS / STRESS TEST  
Interpretation: clumpy chromatic source model can generate large ratios, but loses systematic ordering.

Expected output:
- max ratio can become large
- P(C814>C390) ~0.54

### forward_null_v3_nonlens_controls.py
Status: PASS  
Interpretation: non-lens control fields do not reproduce the observed chromatic lensing structure.

Expected output:
- mean ratio ~1.01
- 99.9% ratio ~1.25
- max ratio ~2.6

## REVIEW: Cosmology consistency scripts

### cmb_blackbody_consistency_test.py
Status: REVIEW  
Interpretation: identifies potential FIRAS blackbody consistency tension. Should be treated as a constraint, not a confirmed positive validation.

### bao_consistency_test.py
Status: REVIEW  
Interpretation: naive BAO projection appears too large. Should be treated as a failed or constrained naive model unless projection suppression is included.
