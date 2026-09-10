# Customer Churn Prediction

An end-to-end machine learning system that predicts customer churn using customer, account, service, and billing information, with model comparison, explainable AI using SHAP, and an interactive Streamlit dashboard.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1+-green.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange.svg)](https://shap.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)]()

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Dataset](#dataset)
5. [Machine Learning Workflow](#machine-learning-workflow)
6. [Models](#models)
7. [Evaluation](#evaluation)
8. [Explainability](#explainability)
9. [Streamlit Dashboard](#streamlit-dashboard)
10. [Project Structure](#project-structure)
11. [Installation](#installation)
12. [Running the Notebook](#running-the-notebook)
13. [Running Streamlit](#running-streamlit)
14. [Model Artifacts](#model-artifacts)
15. [GitHub Files Policy](#github-files-policy)
16. [Limitations](#limitations)
17. [Future Improvements](#future-improvements)
18. [License](#license)

---

## Project Overview

Customer attrition (churn) represents a critical financial challenge for recurring-revenue businesses. In the telecommunications sector, acquiring a replacement customer is estimated to cost **5 to 7 times more** than retaining an existing account.

This project delivers a complete, production-grade Machine Learning solution that predicts and mitigates customer attrition. Built on real-world telecommunications data, the system includes:
- An end-to-end reproducible 32-section Jupyter Notebook covering exploratory analysis, feature engineering, and model development.
- A modular production inference pipeline (`src/pipeline.py`) that eliminates training-serving skew.
- Multi-model benchmarking across **Logistic Regression**, **Random Forest**, and **XGBoost**, tuned with cost-sensitive weighting to target high recall.
- Game-theoretic explainability powered by **TreeSHAP** for both portfolio-wide driver analysis and customer-level waterfall attributions.
- A modern, interactive **Streamlit web application** supporting single-customer risk scoring, localized retention playbooks, and high-throughput batch CSV scoring with portfolio revenue-at-risk analytics.

---

## Problem Statement

Telecommunications subscribers defect to competing providers due to pricing pressures, service dissatisfaction, billing friction, or inflexible contractual commitments. Without predictive intelligence, retention teams operate reactively—attempting to intervene only after a customer submits a cancellation request, a point at which win-back success rates typically fall below 15%.

This system formulates customer churn as a **supervised binary classification** problem under realistic class imbalance (~26.5% churn prevalence). The core challenge is to reliably identify accounts exhibiting pre-churn behavioral patterns while balancing precision to prevent unnecessary promotional discounting on loyal customers.

---

## Objectives

1. **End-to-End Workflow**: Develop a reproducible, end-to-end ML pipeline from data ingestion to interactive deployment.
2. **Feature Engineering**: Engineer domain-specific features capturing customer lifecycle phases, contract stability, service adoption, and billing creep.
3. **Multi-Model Benchmarking**: Evaluate linear and ensemble architectures (Logistic Regression, Random Forest, XGBoost) using 5-fold Stratified Cross-Validation.
4. **Imbalance-Aware Optimization**: Tune hyperparameters using cost-sensitive learning (`scale_pos_weight`) to maximize **Recall** (~80% churn capture) while maintaining high **ROC-AUC (0.846)**.
5. **Model Explainability**: Provide global feature importance and granular local SHAP waterfall explanations for transparent, audited decision-making.
6. **Operational Deployment**: Deliver an executive-grade Streamlit web interface for real-time risk assessment, batch CSV predictions, and business retention strategies.
7. **Production Quality**: Guarantee training-serving consistency, robust schema validation, clean modular design, and strict repository hygiene.

---

## Dataset

* **Dataset Name**: IBM Telco Customer Churn
* **Source**: IBM Developer Sample Datasets / IBM Cognos Analytics
* **Source URL**: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
* **Observations**: 7,043 customer records
* **Attributes**: 21 raw columns (demographics, services, contract terms, billing amounts)
* **Target Variable**: `Churn` (`Yes` = 1, `No` = 0)
* **Class Distribution**:
  - `Retained (No Churn)`: 5,174 accounts (73.46%)
  - `Defected (Churn)`: 1,869 accounts (26.54%)
  - **Class Imbalance Ratio**: **2.77 : 1** (negative to positive)

### Feature Categories
- **Customer Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account & Contract Attributes**: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`
- **Subscribed Telephony & Internet Services**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Billing Details**: `MonthlyCharges`, `TotalCharges`

### Dataset Git Policy & Reproduction
In accordance with repository hygiene policies:
- **Only the cleaned, feature-engineered dataset** (`data/cleaned/cleaned_dataset.csv`) is tracked in this Git repository.
- The raw dataset (`data/raw/Telco-Customer-Churn.csv`) is intentionally excluded via `.gitignore`.
- If you wish to obtain or re-download the raw dataset, execute `python train.py`, which includes automated fallback logic to fetch the original CSV directly from the official IBM repository URL.

---

## Machine Learning Workflow

The end-to-end pipeline follows a disciplined machine learning lifecycle:

```
Raw Dataset
    ↓
Data Cleaning
    ↓
EDA (Exploratory Data Analysis)
    ↓
Feature Engineering
    ↓
Feature Selection
    ↓
Preprocessing
    ↓
Model Training
    ↓
Model Comparison
    ↓
Hyperparameter Tuning
    ↓
Final Model Selection
    ↓
Model Evaluation
    ↓
SHAP Explainability
    ↓
Streamlit Deployment
```

### Lifecycle Highlights
1. **Data Cleaning**: Stripped whitespace in `TotalCharges`, coerced blank strings (11 zero-tenure accounts) to `0.0`, and removed non-predictive identifiers (`customerID`).
2. **Exploratory Data Analysis (EDA)**: Analyzed univariate distributions and bivariate relationships; identified high churn concentrations among month-to-month contracts (42.7%) and fiber optic internet users (41.9%).
3. **Feature Engineering**: Created 7 domain features:
   - `tenure_group`: Lifecycle cohorts (`0-12m`, `13-24m`, `25-48m`, `49-60m`, `>60m`).
   - `total_services`: Integer count (0-8) of active subscribed services.
   - `has_tech_support_or_security`: Binary indicator for protective support services.
   - `is_long_term_contract`: Indicator for 1-year or 2-year contractual commitments.
   - `automatic_payment`: Indicator for automated bank transfer or credit card billing.
   - `monthly_to_total_ratio`: Ratio of current monthly charge to lifetime bill.
   - `avg_monthly_charges_diff`: Difference between current monthly charge and historical average (`MonthlyCharges - TotalCharges / tenure`), detecting price shock and bill creep.
4. **Preprocessing**: Built a Scikit-Learn `ColumnTransformer` with `StandardScaler` for numeric columns and `OneHotEncoder(drop='first', handle_unknown='ignore')` for categorical columns, generating 40 model-ready features.
5. **Cross-Validation & Tuning**: 5-fold Stratified Cross-Validation on training data (80% train / 20% test split, 1,409 holdout accounts). `GridSearchCV` optimization on tree depth, learning rate, subsampling, and positive class weighting.
6. **Artifact Serialization**: Exported canonical feature names (`config/feature_names.json`) and model metadata (`models/model_metadata.json`).

---

## Models

Three distinct model families were evaluated to balance interpretability, variance reduction, and non-linear predictive capacity:

1. **Logistic Regression (Baseline)**:
   - Linear classification baseline trained with `class_weight='balanced'` and L2 regularization (`max_iter=1000`).
   - Offers fully transparent log-odds coefficients and fast convergence.

2. **Random Forest Classifier**:
   - Non-linear bagging ensemble comprising 150 randomized decision trees.
   - Configured with `class_weight='balanced'`, `max_depth=10`, and `min_samples_split=5` to control tree complexity and mitigate overfitting.

3. **XGBoost Classifier (Final Production Model)**:
   - Extreme Gradient Boosting configured with shallow depth trees (`max_depth=3`) and shrinkage (`learning_rate=0.03`, `n_estimators=200`).
   - Subsampling (`subsample=0.8`, `colsample_bytree=0.8`) to prevent co-adaptation of features.
   - Cost-sensitive loss weighting (`scale_pos_weight=2.7686`) to heavily penalize false negatives and maximize churn detection.

---

## Evaluation

All models were evaluated on the independent holdout test set (1,409 accounts, exactly 20% stratified split).

### Benchmark Comparison

| Model | ROC-AUC | PR-AUC | F1-Score | Recall | Precision | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tuned XGBoost** | **0.8460** | **0.6630** | **0.6262** | **0.7995** | **0.5146** | **0.7466** |
| **Logistic Regression** | 0.8427 | 0.6406 | 0.6173 | 0.7914 | 0.5060 | 0.7395 |
| **XGBoost (Baseline)** | 0.8417 | 0.6592 | 0.6286 | 0.7941 | 0.5201 | 0.7509 |
| **Random Forest** | 0.8414 | 0.6518 | 0.6362 | 0.7433 | 0.5560 | 0.7743 |

### Confusion Matrix (Tuned XGBoost at Default 0.50 Threshold)

```
                       Predicted Retained    Predicted Churn
Actual Retained (0)           753                  282
Actual Churn    (1)            75                  299
```

- **True Negatives (TN)**: 753 (Accurately classified loyal accounts)
- **False Positives (FP)**: 282 (Customers offered retention outreach unnecessarily)
- **False Negatives (FN)**: 75 (At-risk accounts missed)
- **True Positives (TP)**: 299 (At-risk accounts correctly captured for intervention)

### Evaluation Analysis
- **Recall = 79.95%**: Captures ~8 out of 10 churning customers, minimizing costly unaddressed defections.
- **ROC-AUC = 0.8460**: High overall discriminatory capability across various operating thresholds.
- **PR-AUC = 0.6630**: Strong baseline precision-recall separation against a 26.5% minority class baseline.

---

## Explainability

Model decisions are interpreted through **SHAP (SHapley Additive exPlanations)**, grounding predictions in cooperative game theory.

### Global Feature Drivers
TreeSHAP analysis across the cohort revealed the key behavioral drivers:
- **Top Risk-Increasing Factors**:
  - `Contract_Month-to-month`: Short commitment window; single strongest driver of churn.
  - `InternetService_Fiber optic`: Higher churn propensity due to price sensitivity and service friction.
  - `PaymentMethod_Electronic check`: Manual billing friction correlates with higher attrition.
  - `MonthlyCharges` & `avg_monthly_charges_diff`: Recent bill increases or higher recurring charges heighten churn risk.
- **Top Retention Anchors (Risk-Decreasing)**:
  - `tenure` & `tenure_group_>60m`: Account longevity strongly cements loyalty.
  - `Contract_Two year` & `Contract_One year`: Contractual commitments effectively eliminate near-term churn.
  - `has_tech_support_or_security`: Subscribed protective services increase platform stickiness.
  - `automatic_payment`: Seamless automated payments reduce transactional friction.

### Local Customer-Level Explanations
In the Streamlit application, individual predictions feature real-time SHAP waterfall charts. Each feature's positive or negative push is quantified relative to the baseline expected value (`E[f(x)]`), providing frontline account managers with transparent rationale for every score.

---

## Streamlit Dashboard

The production Streamlit application (`app.py`) provides an enterprise-ready command center for customer risk intelligence:

1. **Single Customer Risk Assessment**:
   - Input controls for demographics, account settings, subscribed services, and billing details.
   - Instant calculation of **Churn Probability** (0–100%) and **Risk Tier** (`Low < 30%`, `Medium 30–60%`, `High ≥ 60%`).
   - Dynamic SVG risk gauge and status indicators.
   - Interactive local feature attribution waterfall chart.
   - Tailored business retention recommendations (contract conversion discounts, security service bundles, autopay credits).

2. **Batch CSV Prediction & Portfolio Analytics**:
   - Drag-and-drop CSV upload with instant schema validation and missing-value handling.
   - "Download Sample CSV Template" button using pre-formatted customer records.
   - Portfolio KPIs: Total accounts analyzed, predicted churn count, overall churn rate (%), and **Monthly Recurring Revenue at Risk ($)**.
   - Interactive Risk Distribution donut chart and probability distribution histograms.
   - Exportable enriched CSV containing calculated churn probabilities and assigned risk categories.

3. **Model Performance Hub**:
   - Cross-model comparative performance table and hyperparameter specifications.
   - Interactive decision threshold slider to evaluate precision-recall trade-offs.

4. **Dataset Insights & Retention Playbook**:
   - Exploratory visualizations covering tenure, contracts, and payment methods.
   - Strategic retention playbooks for executive decision-makers.

---

## Project Structure

```
Customer-Churn-Prediction/
├── README.md                            # Publication-quality project documentation
├── requirements.txt                     # Pinned Python dependencies
├── app.py                               # Production Streamlit web application
├── customer-churn-prediction.ipynb      # Complete 32-section reproducible Jupyter Notebook
├── train.py                             # Automated training, evaluation & artifact export script
├── test_app_logic.py                    # Automated test & verification suite
│
├── src/                                 # Modular application package
│   ├── __init__.py                      # Package initializer
│   └── pipeline.py                      # Preprocessing, feature engineering & inference pipeline
│
├── data/
│   ├── cleaned/
│   │   └── cleaned_dataset.csv          # Cleaned & feature-engineered dataset (tracked in Git)
│   └── sample_customers.csv             # 20 customer records for batch testing (tracked in Git)
│
├── config/
│   └── feature_names.json               # Canonical 40-feature schema in exact model order (tracked in Git)
│
└── models/
    └── model_metadata.json              # Versioning, schema, parameters & benchmark metrics (tracked in Git)
```

> **Local Project Configuration Note**: A local `.gitignore` is maintained in the development workspace to enforce exclusion rules (preventing raw data, caches, and large model binaries from tracking), but is kept local-only and not committed to the remote repository.

### Key Modules & Files
- `src/pipeline.py`: Implements `ChurnPipeline`, providing identical feature engineering, encoding, and inference logic for both batch and single-record predictions.
- `train.py`: Self-contained script to train the model, fit preprocessing objects, log metrics, and export serialized model binaries.
- `test_app_logic.py`: Verification suite testing data integrity, model loading, high/low risk scenarios, batch predictions, and malformed CSV handling.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ThayyilAnas/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Set Up a Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux / macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Notebook

To explore the full 32-section data science workflow, launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook customer-churn-prediction.ipynb
```

To execute the entire notebook non-interactively from the terminal:

```bash
jupyter nbconvert --to notebook --execute customer-churn-prediction.ipynb --inplace
```

The notebook executes seamlessly from beginning to end, leveraging relative project paths and reproducible random seeds.

---

## Running Streamlit

Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

> **Workflow on a Fresh Clone**:
> Because large binary model files (`models/*.pkl`) are excluded from Git to keep the repository lightweight and portable, follow these steps to run the application from scratch:
> 1. **Run the notebook** (or execute `python train.py` for automated headless training)
> 2. **Generate the model artifacts** (`models/model.pkl`, `models/preprocessor.pkl`, `models/scaler.pkl`)
> 3. **Run Streamlit**:
>    ```bash
>    streamlit run app.py
>    ```

---

## Model Artifacts

To maintain repository speed and adhere to GitHub best practices, serialized binaries are separated from configuration metadata:

| Artifact | Location | Git Status | Description |
| :--- | :--- | :---: | :--- |
| `feature_names.json` | `config/` | **Tracked** | 40 canonical encoded features in exact model order |
| `model_metadata.json` | `models/` | **Tracked** | Model hyperparameters, training dates, and benchmark metrics |
| `model.pkl` | `models/` | **Ignored** | Serialized Tuned XGBoost production model binary (generated locally) |
| `preprocessor.pkl` | `models/` | **Ignored** | Serialized Scikit-Learn `ColumnTransformer` (generated locally) |
| `scaler.pkl` | `models/` | **Ignored** | Serialized Scikit-Learn `StandardScaler` (generated locally) |

All ignored model binaries are automatically reproduced by running `python train.py`.

---

## GitHub Files Policy

To ensure clean repository hygiene, this project strictly separates tracked source assets from transient runtime outputs.

### Included (Tracked in Git)
- **Application Source Code**: `app.py`, `train.py`, `test_app_logic.py`, and `src/` modules.
- **Jupyter Notebook**: Fully executed, clean `customer-churn-prediction.ipynb`.
- **Cleaned Dataset**: Cleaned, feature-engineered dataset (`data/cleaned/cleaned_dataset.csv`).
- **Sample Data**: Batch prediction test template (`data/sample_customers.csv`).
- **Configuration & Metadata**: `config/feature_names.json` and `models/model_metadata.json`.
- **Project Configuration**: `requirements.txt` and `README.md`.

### Excluded (Ignored via `.gitignore`)
- **Raw Datasets**: `data/raw/Telco-Customer-Churn.csv` and temporary archive formats (`*.zip`, `*.tar.gz`).
- **Large Model Binaries**: Serialized models (`models/*.pkl`, `models/*.joblib`).
- **Virtual Environments**: `.venv/`, `venv/`, `env/`.
- **Python Cache**: `__pycache__/`, `*.pyc`, `*.pyo`.
- **Jupyter Checkpoints**: `.ipynb_checkpoints/`.
- **IDE Settings**: `.vscode/`, `.idea/`.
- **OS-Generated Files**: `.DS_Store`, `Thumbs.db`.
- **Generated Output Charts & Reports**: `outputs/`, `reports/`, `*.pdf`, `*.png`, `*.jpg`.
- **Temporary Scripts**: One-off generator or utility scripts (`capture_all_views.py`, `create_pdf.py`, etc.).

---

## Limitations

1. **Correlation vs. Causation**: SHAP values explain statistical associations learned from observational data, not counterfactual causation. Retention initiatives (such as promotional price discounts) should be tested via randomized controlled trials (A/B testing) to measure true incrementality.
2. **Cross-Sectional Nature**: The dataset reflects an aggregated snapshot rather than longitudinal time-series logs (e.g., daily call quality, network outages, real-time ticket escalation history).
3. **Threshold Sensitivity**: The default 0.50 decision threshold prioritizes high recall (~80%), which incurs false positives. For organizations operating under fixed retention budgets, the classification threshold should be tuned upward (e.g., 0.60–0.65).
4. **Domain Specificity**: The model is optimized for telecommunications subscription dynamics and should not be applied to contractual B2B SaaS or transactional retail without domain retraining.

---

## Future Improvements

* **Survival Analysis**: Implement Cox Proportional Hazards or Kaplan-Meier models to estimate the expected *time-to-churn* rather than just a static binary probability.
* **Customer Lifetime Value (CLV) Integration**: Combine churn risk with forward-looking customer lifetime value to optimize retention budget allocation based on net dollar impact.
* **Uplift Modeling**: Train causal uplift models to segment customers into *Persuadables*, *Sure Things*, *Lost Causes*, and *Sleeping Dogs*.
* **Streaming Telemetry Ingestion**: Connect real-time streaming pipelines (e.g., Kafka / Apache Flink) to score accounts dynamically after negative service events (e.g., repeated tech support calls).
* **Automated MLOps Pipeline**: Implement CI/CD retraining workflows with MLflow / DVC for automated model registry, experiment tracking, and data drift monitoring.

---

## License

License: Not specified.
