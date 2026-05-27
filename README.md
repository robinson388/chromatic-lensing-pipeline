# Chromatic Lensing Pipeline

Reproducible analysis pipeline for chromatic directional-coherence measurements in strong gravitational lenses.

This repository contains scripts, null tests, Monte Carlo outputs, and processed CSV results associated with wavelength-dependent coherence anomalies in HST/JWST/ALMA lensing data.

## Structure

- `scripts/core/` — directional coherence and chromatic ordering pipelines
- `scripts/null_tests/` — achromatic GR, PSF, clumpy-source, and non-lens null simulations
- `scripts/geometry/` — radial shift, torsion, and cosmic-prism geometry analyses
- `scripts/cosmology/` — Cdir-to-H0, BCMB, BAO, and blackbody consistency tests
- `scripts/validation/` — CASTLES, external lens, and radio validation tools
- `results/` — processed CSV outputs and Monte Carlo summaries

## Data note

Raw FITS files are not included by default because of size and external archive provenance.
Scripts expect local HST/JWST/ALMA FITS data arranged under `data/`.

## Author

Robinson Bueno Parra
