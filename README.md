# IDD: Information-Geometric Distribution Detection

Code for the ICML accepted paper *"Beyond Euclidean Summaries: Online Change Point Detection for
Distribution-Valued Data"*.

Paper: https://arxiv.org/abs/2602.07252


---

## Repository Structure

```
github_submission/
├── idd_core/                    # Shared IDD algorithm implementations
│   ├── ot_mfpca_flow.py         # OT tangent maps + mFPCA (flow-based barycenter, high d)
│   ├── ot_mfpca_empirical.py    # OT tangent maps + mFPCA (empirical OT, low d)
│   └── ot_mfpca_discrete_core.py# IDD for discrete supports (Hamming / ordinal cost)
│
├── continuous_streams/          # Sec 5.1 – Continuous stream experiments
│   ├── data_generation/
│   │   ├── generate_continuous_data.py  # Simulate IC / mm_reweight / copula_shift / barycenter
│   │   └── common_ot.py                 # OT primitive helpers (barycenter, maps)
│   ├── baselines.py             # Log-KDE-MFPCA + Hotelling T² baselines
│   ├── baselines_cpd.py         # F-CPD (Fréchet), NEWMA, Scan-B baselines
│   ├── common_mfpca.py          # R/funcharts mFPCA interface
│   ├── run_mfpca.R              # R script called internally
│   ├── main_run_flow.py         # Main experiment runner (all methods, all scenarios)
│   ├── run_all_simu.py          # Batch launcher for N ∈ {50, 100, 300}
│   └── summary_figs.py          # ARL₀ vs ARL₁ trade-off plots (Fig. 1)
│
├── discrete_streams/            # Sec 5.2 – Discrete stream experiments
│   ├── data_generation/
│   │   └── generate_discrete_data.py    # counting (Poisson spike / zero-infl), dust-on-screens, categorical
│   ├── main_run_discrete_all.py         # Experiment runner (attribute chart baselines)
│   └── tradeoff_discrete_v3.py          # Trade-off plots (Fig. 2)
│
├── gaussian_translation/        # Sec 5.3 / Theorem F.1 – IDD vs log-KDE on Gaussian shift
│   ├── data_generation/
│   │   └── generate_gaussian_data.py   # N(m,Σ) → N(m+δ/√n, Σ) with replications
│   ├── ot_mfpca.py                     # OT processing for Gaussian comparison
│   ├── log_kde.py                      # Log-KDE-MFPCA implementation
│   ├── ot_mfpca_once.R                 # R mFPCA wrapper for this experiment
│   └── run_scripts_all.py              # Full experiment runner (Table 1)
│
└── case_study/                  # Sec 6 – Reddit COVID-vaccine monitoring
    ├── data_processing.py       # Daily windows + sentiment3d / embed_pca20 features
    ├── run_reddit_vax.py        # Main runner: OT-MFPCA + KDE + Hotelling T² on daily streams
    └── visualization_cpd.py     # Multi-panel Phase-II monitoring figure (IDD vs CPD baselines)
```

---

## Setup

**Python** ≥ 3.10 and **R** ≥ 4.2 with the `funcharts` package.

```bash
pip install -r requirements.txt

# In R:
install.packages("funcharts")
```

---

## Reproducing the Paper Results

### 1. Continuous Streams

**Step 1 – Generate data** (or set `DIDO_DATA_ROOT` to an existing data folder).
Generate each batch size you plan to run; outputs are stored as `continuous/n<N>/`:
```bash
cd continuous_streams/data_generation
python generate_continuous_data.py --out ../../data/continuous --n_points 50
python generate_continuous_data.py --out ../../data/continuous --n_points 100
python generate_continuous_data.py --out ../../data/continuous --n_points 300
```

**Step 2 – Run experiments**:
```bash
cd continuous_streams
# Run all three generated batch sizes: N=50, 100, 300
python run_all_simu.py
```
Or for a single batch size:
```bash
python main_run_flow.py 100
```

**Step 3 – Generate trade-off plots**:
```bash
python summary_figs.py
```

> **Environment variable**: set `DIDO_DATA_ROOT` to override the default `../data/` path.

---

### 2. Discrete Streams 

**Step 1 – Generate data**:
```bash
cd discrete_streams/data_generation
python generate_discrete_data.py --n_points 100
```

**Step 2 – Run experiment** (IDD + attribute chart baselines):
```bash
cd discrete_streams
python main_run_discrete_all.py 100
```

**Step 3 – Trade-off plots**:
```bash
python tradeoff_discrete_v3.py
```

---

### 3. Gaussian Translation Analysis
**Step 1 – Generate data**:
```bash
cd gaussian_translation/data_generation
python generate_gaussian_data.py
```

**Step 2 – Run IDD vs log-KDE comparison**:
```bash
cd gaussian_translation
python run_scripts_all.py
```

---

### 4. Case Study – Reddit COVID-vaccine Monitoring

The dataset is `SummaryResults_Covid_All.csv` from the [COVID-19 vaccine Reddit corpus](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XJTBQM).
Set the path via environment variable or `--csv` argument.

**Step 1 – Preprocess weekly windows**:
```bash
cd case_study
python data_processing.py --csv /path/to/SummaryResults_Covid_All.csv --phase1_end jj_eua
```

**Step 2 – Run monitoring**:
```bash
export DIDO_REDDIT_CSV=/path/to/SummaryResults_Covid_All.csv
python run_reddit_vax.py
```

Phase I: last 50 daily windows before the J&J EUA (2021-02-27).  
Phase II: all daily windows from the J&J EUA onward.

---


## Citation

```bibtex
@inproceedings{zeng2026idd,
  title     = {IDD: Sequential Distribution Monitoring via Optimal Transport and Functional PCA},
  author = {Zeng, Yingyan and Huang, Yujing (Zipan) and Chen, Xiaoyu},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
}
```
