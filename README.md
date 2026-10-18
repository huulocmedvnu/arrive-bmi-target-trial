# Delivery at 39 weeks versus expectant management and perinatal death across maternal body mass index

Analysis code for a target trial emulation of delivery in week 39 compared with expectant management in low-risk,
nonanomalous singleton pregnancies ongoing at 39 0/7 weeks, United States, 2019 to 2024, by prepregnancy body mass
index. The data are the public-use natality, fetal death and linked birth/infant death period files of the National
Center for Health Statistics (NCHS). No individual-level data are included in this repository.

## Contents

| Path | Purpose |
|---|---|
| `rust/` | `vitals`: multithreaded fixed-width parser of the NCHS files. Applies the eligibility criteria and writes counts by covariate profile (`ps`, `tte`), the study flow (`flow`) and birthweight distributions (`bwq`). Field positions follow the NCHS user guides. |
| `rust_ccw/` | `ccw`: clone-censor-weight estimation. Logistic models for the probabilities of induction and birth in week 39 and of continuation to 40 0/7 weeks within each BMI class, weights truncated at the 99th percentile, normalized risks, and a Bayesian bootstrap that refits both models in every draw. |
| `rust_bayes/` | `bayes`: No-U-Turn sampler (nuts-rs) for the Bayesian Poisson model of the conventional comparison. |
| `scripts/download*.sh` | Download the NCHS files and user guides. |
| `scripts/ccw.py` | Primary analysis, gradient, E-values, bias analysis, balance, population impact and sensitivity analyses. Calls `rust_ccw`. |
| `scripts/ccw_sub.py` | Subgroups, 2-unit BMI bins and natural cubic spline. |
| `scripts/iptw.py`, `scripts/tte.py`, `scripts/tte_summary.py`, `scripts/bayes_export.py` | Conventional comparisons (propensity weighting, Bayesian Poisson model). |
| `scripts/bw_thresholds.py` | Birthweight 10th percentile by week and sex. |
| `scripts/nd_unmatched.py` | Diagnostic for linked neonatal deaths without a covariate profile at risk. |
| `scripts/manuscript_numbers.py`, `scripts/figures_ms.py` | Tables, reported numbers and figures. |
| `scripts/validate_rust.py`, `scripts/check_draw.py`, `scripts/compare_boot.py` | Checks of `rust_ccw` against statsmodels on identical data and bootstrap draws. |
| `scripts/run_all.sh` | Whole pipeline. |

## Running

Requirements: Rust (tested with 1.94), Python 3.13 with `requirements.txt`, and for the convergence diagnostics of the
Bayesian Poisson model a Python environment with `requirements-bayes.txt`. About 6 GB of downloads.

```bash
PY=python PY_BAYES=/path/to/bayes/python NCPU=12 bash scripts/run_all.sh
```

Results are written to `outputs/` (JSON and TSV) and `figures/manuscript/`. With 48 threads, the 1,000-draw refitted
bootstrap of the primary analysis takes about 20 seconds. Random seeds are fixed in the scripts.

## Validation

`rust_ccw` was checked against statsmodels generalized linear models on the same covariate-profile counts and the same
bootstrap draws (`scripts/validate_rust.py`, `scripts/check_draw.py`): coefficients agreed to about 2e-8, the
convergence tolerance of statsmodels, and weighted risks to below 1e-9. The Rust crates include unit tests
(`cargo test --release`).

## License

MIT. See `LICENSE`.
