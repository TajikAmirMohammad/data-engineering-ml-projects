<div align="center">

# End-to-End Production-Grade Fraud Detection Pipeline
### Cost-Sensitive Machine Learning, Decision Threshold Calibration & Explainable AI (XAI)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB5424?logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/SHAP-Interpretability-brightgreen)](https://shap.readthedocs.io/)

<p align="center">
  A resilient, leakage-free financial fraud detection framework designed for severe class imbalance (0.173% fraud prevalence). Optimized using real-world banking cost matrices to cut financial misclassification loss by <b>34.42%</b> while maintaining regulatory compliance (GDPR Article 22) through TreeSHAP attributions.
</p>

</div>

---

## 📌 Executive Summary

Traditional fraud detection systems frequently fail in production due to the **Accuracy Paradox**, arbitrary classification thresholds ($T = 0.5$), and lack of post-hoc interpretability. In financial systems, the penalty for a False Negative (missed fraud incurring direct chargeback liabilities) dwarfs the cost of a False Positive (operational triage overhead).

This repository houses a modular, 6-phase machine learning pipeline evaluated on **284,807 European card transactions**. Rather than optimizing for raw accuracy, this framework aligns model training and decision boundaries directly with business unit economics, achieving high recall coverage while drastically controlling manual review burdens.

### 🎯 Key Performance Highlights
- **Direct Financial Impact:** **34.42% reduction** in total financial loss compared to standard $T=0.5$ baselines.
- **Calibrated Operational Boundary:** Shifted decision threshold to **$T^* = 0.2070$** to optimize the asymmetric cost function ($C_{FN}=\$150$, $C_{FP}=\$5$).
- **Fraud Coverage:** Achieved **88.78% Fraud Recall** on quarantined test data, reducing missed incidents to just 11 transactions out of 56,962 test records.
- **Production-Grade Data Hygiene:** Enforced strict temporal and cross-validation boundaries with **zero data leakage**.
- **Regulatory Governance:** Decoded black-box predictions via **TreeSHAP** to meet regulatory requirements (GDPR Article 22 & ECOA compliance).

---

## 🏗️ 6-Phase Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PIPELINE WORKFLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘
  [Phase 1] Data Ingestion & Quality Audit (Zero NaNs, 0.173% Imbalance)
      │
      ▼
  [Phase 2] Leakage-Free Preprocessing & Feature Engineering
            ├── Stratified Split (80/20) Isolation Barrier
            ├── Cyclical Diurnal Encoding [sin/cos(2π·Hour/24)]
            ├── Domain Feature: MAD-derived `Is_Extreme_Amount`
            └── RobustScaler on Transaction Amount
      │
      ▼
  [Phase 3] Cost-Sensitive Modeling & Multi-Model Benchmarking
            ├── Logistic Regression (Class-Weighted)
            ├── Random Forest (Subsample-Balanced) ──► Champion Model (PR-AUC: 0.8565)
            └── XGBoost (Scale-Pos-Weight Adjusted)
      │
      ▼
  [Phase 4] Stratified Hyperparameter Optimization (5-Fold CV via PR-AUC)
      │
      ▼
  [Phase 5] Financial Cost Matrix & Decision Threshold Calibration (T* = 0.2070)
      │
      ▼
  [Phase 6] Explainability & Regulatory Auditing (TreeSHAP Summary & Local Waterfall)
```

---

## 🔬 Technical Methodology by Phase

### Phase 1: Ingestion & Telemetry Profiling
- Verified dataset integrity across 31 continuous dimensions. Confirmed **0 missing values**.
- Detected 1,081 duplicate transactions ($0.38\%$), which were deliberately preserved to reflect legitimate payment retry mechanisms and automated bot patterns.
- Diagnosed extreme class asymmetry: **99.827% Legitimate vs. 0.173% Fraud** via dual-scale (linear vs. logarithmic) distribution profiling.

### Phase 2: Feature Engineering & Anti-Leakage Protocol
- **Stratified Partitioning:** Quarantined an exact 20% evaluation partition ($N=56,962$) with strictly verified zero index overlap.
- **Trigonometric Diurnal Embedding:** Mapped cumulative transaction seconds into cyclical 24-hour sinusoidal coordinates (`Hour_sin`, `Hour_cos`), removing the synthetic 23:59 $\rightarrow$ 00:00 step boundary.
- **Robust Anomaly Flagging:** Calculated an empirical upper-bound cutoff ($~\$104.90$) strictly on $X_{train}$ using Median Absolute Deviation (MAD), generating the binary heuristic `Is_Extreme_Amount`.
- **Interquartile Scaling:** Applied `RobustScaler` (Median/IQR) to continuous volumes to mitigate extreme skewness without clipping fraud spikes.

### Phase 3 & 4: Model Benchmarking & Stratified Tuning
Models were evaluated using **PR-AUC (Average Precision)** as the primary target metric, avoiding optimistic distortions common to ROC-AUC under extreme class skews.

| Model Candidate | PR-AUC | ROC-AUC | Recall (%) | Precision (%) | False Positives (FP) | Missed Fraud (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | 0.7326 | 0.9813 | **91.84%** | 6.21% | 1,359 | **8** |
| **XGBoost (Cost-Sensitive)** | 0.8401 | 0.9849 | 86.73% | 58.62% | 60 | 13 |
| **Random Forest (Champion)** | **0.8565** | **0.9852** | 82.65% | **84.38%** | **15** | 17 |

*Random Forest achieved the strongest latent space ranking power while reducing false alarms from 1,359 to just 15.*
Using 5-Fold `StratifiedKFold` randomized search, optimal tree regularization parameters were locked in (`n_estimators=80`, `max_depth=12`, `min_samples_leaf=2`, `class_weight='balanced_subsample'`).

---

## 💰 Phase 5: Financial Loss Minimization

In real-world deployment, setting a threshold at $0.5$ assumes symmetric risk. We modeled the total operational financial loss function:

$$\text{Total Loss}(T) = \big(C_{\text{FN}} \times \text{FN}(T)\big) + \big(C_{\text{FP}} \times \text{FP}(T)\big)$$

* **$C_{\text{FN}} = \$150$**: Direct chargeback reimbursement, network fines, processing penalties.
* **$C_{\text{FP}} = \$5$**: Operational review labor, SMS/email verification dispatch.

```
       FINANCIAL THRESHOLD OPTIMIZATION CURVE
 Loss ($)
   ▲
$3k│           Default T=0.50 ($2,905.00)
   │             \
$2k│              \   Optimal T* = 0.2070 ($1,905.00)
   │               \   |
$1k│                └──*─────────── Savings: $1,000 (34.42%)
   └────────────────────────────────────────────────► Threshold (0 to 1)
```

### Operational Metrics at Optimal Threshold ($T^* = 0.2070$):
* **Default Threshold Cost ($T=0.50$):** $\$2,905.00$
* **Optimized Minimum Cost ($T=0.2070$):** $\$1,905.00$
* **Direct Cost Reduction:** **$\$1,000.00$ ($34.42\%$ operational savings)**
* **Fraud Recall at $T^*$:** Increased from $82.65\%$ to **$88.78\%$** (Capturing 87 of 98 fraud cases).

---

## 🔍 Phase 6: Explainability & Governance (TreeSHAP)

To fulfill **GDPR Article 22 (Right to an Explanation)** and prevent unmonitored algorithmic bias, the champion model was coupled with an optimized `TreeExplainer`:

1. **Global Attribution Dynamics:** Features `V12`, `V14`, `V4`, `V10`, and `V17` dominate the decision manifold. Extreme negative shifts in `V12` and `V14` provide the strongest positive vectors toward fraud classification.
2. **Local Sample Attribution:** Individual transactions receive audit-ready waterfall decompositions showing exact baseline-to-decision attribution paths (e.g., verifying an anomalous drop in `V12` pushed risk confidence from base rate to $>98\%$).

---

## 📁 Repository Structure

```plaintext
├── data/
│   └── creditcard.csv              # Source transaction dataset
├── processed_data/
│   ├── X_train.parquet             # Transformed train feature matrix
│   ├── X_test.parquet              # Quarantined test feature matrix
│   ├── y_train.parquet             # Target training vector
│   └── y_test.parquet              # Target evaluation vector
├── notebooks/
│   └── fraud_pipeline.ipynb        # Complete 6-Phase interactive notebook
├── requirements.txt                # Python runtime dependencies
├── LICENSE                         # MIT License
└── README.md                       # Architecture & execution guide
```

---

## 🚀 Getting Started

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` inside the `data/` directory.

### 3. Execution
Launch the pipeline via Jupyter or execute directly:
```bash
jupyter notebook notebooks/fraud_pipeline.ipynb
```

---

## 🛠️ Tech Stack & Tooling
- **Core:** Python, NumPy, Pandas, Scipy
- **Machine Learning:** Scikit-Learn, XGBoost, Joblib
- **Model Explainability:** SHAP (Shapley Additive exPlanations)
- **Data Engineering:** Parquet (Fast columnar persistence)
- **Visualization:** Matplotlib, Seaborn

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).