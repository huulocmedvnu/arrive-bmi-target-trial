#!/usr/bin/env bash
# Full pipeline from the NCHS public-use files to outputs/ and figures/manuscript/.
# PY: Python with requirements.txt. PY_BAYES: Python with requirements-bayes.txt (NumPyro diagnostics only).
# NCPU: parallel file jobs. Set SKIP_DOWNLOAD=1 if inputs/ is already filled.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}; PY_BAYES=${PY_BAYES:-$PY}; NCPU=${NCPU:-12}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p inputs/natality inputs/fetal inputs/linked inputs/docs work outputs figures/manuscript
[ -n "${SKIP_DOWNLOAD:-}" ] || { bash scripts/download.sh; bash scripts/download_linked.sh; }
for c in rust rust_ccw rust_bayes; do (cd $c && cargo build --release -q) || exit 1; done

FILES="inputs/natality/Nat2019us.zip inputs/natality/Nat2020us.zip inputs/natality/Nat2021us.zip inputs/natality/Nat2022us.zip inputs/natality/Nat2023us.zip inputs/natality/Nat2024us.zip inputs/fetal/Fetal2019US.zip inputs/fetal/Fetal2020US.zip inputs/fetal/Fetal2021US.zip inputs/fetal/Fetal2022US.zip inputs/fetal/Fetal2023US.zip inputs/fetal/Fetal2024US.zip"
LINKED="2019PE2018CO.zip VS19LINK.Public.USNUMPUB_R2021_08_30 2020PE2019CO.zip VS20LINK.Public.USNUMPUB_R2022_05_03 2021PE2020CO.zip VS2021LINK.Public.USNUMPUB_R2023_05_31 2022PE2021CO.zip VS2022LINK.Public.USNUMPUB_R2024_08_07 2023PE2022CO.zip VS2023LINK.Public.USNUMPUB_R2024_11_27 2024PE2023CO.zip VS2024LINK.Public.USNUMPUB_R2026_01_14"
export V=rust/target/release/vitals

# vital records: $1 = mode, $2 = output directory
records() { mkdir -p $2; export M=$1 O=$2; printf "%s\n" $FILES | xargs -P $NCPU -I{} sh -c 'f={}; k=nat; case $f in *Fetal*) k=fet;; esac; unzip -p $f | $V $k 8 $M > $O/$(basename $f .zip).tsv 2>/dev/null'; }
# linked infant death numerators: $1 = mode, $2 = output directory
linked() { mkdir -p $2; export M=$1 O=$2; printf "%s\n" $LINKED | xargs -P 6 -n 2 sh -c 'unzip -p inputs/linked/$0 "$1" | $V inf 4 $M > $O/inf_$1.tsv 2>/dev/null'; }

# 1. study flow
records flow work/agg_flow; cat work/agg_flow/*.tsv | grep ^FLOW > outputs/flow_counts.tsv
# 2. birthweight 10th percentiles by week and sex (small for gestational age)
mkdir -p work/agg_bwq
printf "%s\n" $FILES | grep natality | xargs -P $NCPU -I{} sh -c 'unzip -p {} | $V nat 8 bwq > work/agg_bwq/$(basename {} .zip).tsv 2>/dev/null'
cat work/agg_bwq/*.tsv | grep ^BWQ > work/bwq.tsv; $PY scripts/bw_thresholds.py
export BWQ=work/bw_p10.tsv
# 3. covariate-profile counts for the target trial emulation
records ps work/agg_ps; linked ps work/agg_ps; cat work/agg_ps/*.tsv > work/counts_ps.tsv
records tte work/agg_tte; linked tte work/agg_tte; cat work/agg_tte/*.tsv > work/counts_tte.tsv
# 4. conventional comparisons: Bayesian Poisson model (Rust NUTS) and propensity weighting
$PY -W ignore scripts/tte.py
mkdir -p work/bayes/tte/draws
rust_bayes/target/release/bayes 2000 1000 4 work/bayes/tte/draws work/bayes/tte/tte__primary.json work/bayes/tte/tte__no_sb39.json
$PY_BAYES -W ignore scripts/tte_summary.py
(cd scripts && OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 $PY -W ignore iptw.py > /dev/null)
# 5. clone-censor-weight analysis (refitted Bayesian bootstrap in rust_ccw), subgroups, BMI curve
(cd scripts && $PY -W ignore ccw.py && $PY -W ignore ccw_sub.py && $PY -W ignore nd_unmatched.py)
# 6. tables, numbers and figures
$PY scripts/manuscript_numbers.py > /dev/null && $PY scripts/figures_ms.py
echo "done: outputs/ and figures/manuscript/"
