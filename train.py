"""
Complete Training and Artifact Generation Script for Customer Churn Prediction.
Trains Logistic Regression, Random Forest, and XGBoost with Stratified 5-Fold CV,
performs hyperparameter tuning, generates SHAP explainability, and saves production artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import shap

from src.pipeline import (
    clean_raw_data, engineer_features, build_preprocessor,
    extract_feature_names, ALL_MODEL_FEATURES, NUMERIC_FEATURES
)

# 1. Set Seeds and Paths
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

RAW_DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
CLEANED_DATA_PATH = "data/cleaned/cleaned_dataset.csv"
MODELS_DIR = "models"
CONFIG_DIR = "config"
OUTPUTS_DIR = "outputs"

os.makedirs("data/cleaned", exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "evaluation"), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "shap"), exist_ok=True)

print("="*60)
print("1. LOADING AND CLEANING RAW DATASET")
print("="*60)
if os.path.exists(RAW_DATA_PATH):
    raw_df = pd.read_csv(RAW_DATA_PATH)
elif os.path.exists(CLEANED_DATA_PATH):
    print(f"Local raw dataset not found; loading from {CLEANED_DATA_PATH}...")
    raw_df = pd.read_csv(CLEANED_DATA_PATH)
else:
    print("Downloading dataset from official IBM repository...")
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    raw_df = pd.read_csv(url)

print(f"Raw shape: {raw_df.shape}")

cleaned_df = clean_raw_data(raw_df, is_training=True)
print(f"Blank TotalCharges handled: {cleaned_df['TotalCharges'].isnull().sum()} nulls remaining")
print(f"Target Churn distribution:\n{cleaned_df['Churn'].value_counts(normalize=True)}")

print("\n2. FEATURE ENGINEERING")
engineered_df = engineer_features(cleaned_df)
print(f"Engineered dataset shape: {engineered_df.shape}")

# Save cleaned & engineered dataset
engineered_df.to_csv(CLEANED_DATA_PATH, index=False)
print(f"Saved cleaned & engineered dataset to {CLEANED_DATA_PATH}")

# 3. GENERATE EDA FIGURES
print("\n3. GENERATING EDA FIGURES")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Churn Distribution
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x='Churn', hue='Churn', data=raw_df, palette=['#2ecc71', '#e74c3c'], ax=ax, legend=False)
ax.set_title('Overall Customer Churn Distribution (Yes / No)', fontsize=13, fontweight='bold')
ax.set_xlabel('Churn Status', fontsize=11)
ax.set_ylabel('Number of Customers', fontsize=11)
for p in ax.patches:
    ax.annotate(f'{p.get_height()} ({p.get_height()/len(raw_df)*100:.1f}%)',
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "figures", "churn_distribution.png"), dpi=200)
plt.close()

# Churn vs Contract
fig, ax = plt.subplots(figsize=(8, 4.5))
contract_churn = pd.crosstab(raw_df['Contract'], raw_df['Churn'], normalize='index') * 100
contract_churn.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'], ax=ax)
ax.set_title('Churn Rate by Contract Type (%)', fontsize=13, fontweight='bold')
ax.set_ylabel('Percentage of Customers', fontsize=11)
ax.set_xlabel('Contract Term', fontsize=11)
plt.xticks(rotation=0)
plt.legend(['Retained (No)', 'Churned (Yes)'], loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "figures", "churn_by_contract.png"), dpi=200)
plt.close()

# Churn vs Tenure Boxplot
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(x='Churn', y='tenure', hue='Churn', data=cleaned_df, palette=['#2ecc71', '#e74c3c'], ax=ax, legend=False)
ax.set_title('Customer Tenure (Months) vs Churn Status', fontsize=13, fontweight='bold')
ax.set_xlabel('Churn (0 = No, 1 = Yes)', fontsize=11)
ax.set_ylabel('Tenure (Months)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "figures", "churn_vs_tenure.png"), dpi=200)
plt.close()

# Churn vs Monthly Charges
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.kdeplot(cleaned_df[cleaned_df['Churn'] == 0]['MonthlyCharges'], label='Retained (No)', fill=True, color='#2ecc71', alpha=0.4)
sns.kdeplot(cleaned_df[cleaned_df['Churn'] == 1]['MonthlyCharges'], label='Churned (Yes)', fill=True, color='#e74c3c', alpha=0.4)
ax.set_title('Monthly Charges Distribution by Churn Status', fontsize=13, fontweight='bold')
ax.set_xlabel('Monthly Charges ($)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "figures", "churn_vs_monthly_charges.png"), dpi=200)
plt.close()

print("EDA plots saved to outputs/figures/")

# 4. TRAIN / TEST SPLIT & PREPROCESSING
print("\n4. STRATIFIED TRAIN / TEST SPLIT & PREPROCESSING")
X = engineered_df[ALL_MODEL_FEATURES]
y = engineered_df['Churn']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"Training set: {X_train_raw.shape}, Test set: {X_test_raw.shape}")
print(f"Train churn rate: {y_train.mean():.4f}, Test churn rate: {y_test.mean():.4f}")

# Fit ColumnTransformer ONLY on Training Data
preprocessor = build_preprocessor()
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

feature_names = extract_feature_names(preprocessor)
print(f"Total encoded features: {len(feature_names)}")

# Extract fitted scaler for the numeric subset to satisfy models/scaler.pkl
scaler = preprocessor.named_transformers_['num'].named_steps['scaler']

# Save Preprocessor and Scaler immediately
joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
with open(os.path.join(CONFIG_DIR, "feature_names.json"), "w") as f:
    json.dump(feature_names, f, indent=2)
print("Saved preprocessor.pkl, scaler.pkl, and feature_names.json")

# 5. MODEL TRAINING & COMPARISON
print("\n5. MULTI-MODEL CROSS-VALIDATION & BENCHMARKING")
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_split=5,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos, eval_metric='logloss',
        random_state=RANDOM_STATE, n_jobs=-1
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    'roc_auc': 'roc_auc',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'accuracy': 'accuracy',
    'pr_auc': 'average_precision'
}

cv_results = {}
for name, model in models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_results[name] = {
        'ROC-AUC': f"{scores['test_roc_auc'].mean():.4f} +/- {scores['test_roc_auc'].std():.4f}",
        'PR-AUC': f"{scores['test_pr_auc'].mean():.4f} +/- {scores['test_pr_auc'].std():.4f}",
        'F1-Score': f"{scores['test_f1'].mean():.4f} +/- {scores['test_f1'].std():.4f}",
        'Recall': f"{scores['test_recall'].mean():.4f} +/- {scores['test_recall'].std():.4f}",
        'Precision': f"{scores['test_precision'].mean():.4f} +/- {scores['test_precision'].std():.4f}",
        'Accuracy': f"{scores['test_accuracy'].mean():.4f} +/- {scores['test_accuracy'].std():.4f}",
        '_raw': scores
    }
    print(f"--- {name} (5-Fold CV) ---")
    print(f"ROC-AUC: {cv_results[name]['ROC-AUC']} | F1: {cv_results[name]['F1-Score']} | Recall: {cv_results[name]['Recall']}")

# 6. HYPERPARAMETER TUNING
print("\n6. HYPERPARAMETER TUNING (XGBOOST)")
xgb_param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.03, 0.05, 0.1],
    'n_estimators': [100, 150, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [float(scale_pos), float(scale_pos * 0.8)]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1),
    param_grid=xgb_param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)
best_xgb = grid_search.best_estimator_
print(f"Best XGBoost Params: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

# 7. FINAL MODEL EVALUATION ON TEST SET
print("\n7. FINAL TEST SET EVALUATION")
fitted_models = {}
test_evaluations = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model

fitted_models['Tuned XGBoost'] = best_xgb

for name, model in fitted_models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    pr = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    test_evaluations[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc,
        'PR-AUC': pr,
        'ConfusionMatrix': cm.tolist(),
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    print(f"\nModel: {name}")
    print(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | F1: {f1:.4f} | Recall: {rec:.4f} | Precision: {prec:.4f} | Accuracy: {acc:.4f}")

# Model comparison table
comparison_df = pd.DataFrame([
    {
        'Model': k,
        'ROC-AUC': v['ROC-AUC'],
        'PR-AUC': v['PR-AUC'],
        'F1-Score': v['F1-Score'],
        'Recall': v['Recall'],
        'Precision': v['Precision'],
        'Accuracy': v['Accuracy']
    }
    for k, v in test_evaluations.items()
]).sort_values('ROC-AUC', ascending=False)
print("\n=== TEST SET MODEL COMPARISON ===")
print(comparison_df.to_string(index=False))

# Select Final Production Model: Tuned XGBoost
final_model = best_xgb
final_model_name = "Tuned XGBoost"

# Save Production Model Artifact
joblib.dump(final_model, os.path.join(MODELS_DIR, "model.pkl"))
print(f"\nSaved production model ({final_model_name}) to models/model.pkl")

# 8. GENERATE EVALUATION CHARTS
print("\n8. GENERATING EVALUATION CHARTS")
# ROC Curves
plt.figure(figsize=(8, 6))
colors = {'Logistic Regression': '#3498db', 'Random Forest': '#2ecc71', 'XGBoost': '#9b59b6', 'Tuned XGBoost': '#e67e22'}
for name, eval_dict in test_evaluations.items():
    fpr, tpr, _ = roc_curve(y_test, eval_dict['y_proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {eval_dict['ROC-AUC']:.3f})", color=colors.get(name, '#333333'), lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance (AUC = 0.500)')
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "evaluation", "roc_curves.png"), dpi=200)
plt.close()

# Precision-Recall Curves
plt.figure(figsize=(8, 6))
for name, eval_dict in test_evaluations.items():
    precision_pts, recall_pts, _ = precision_recall_curve(y_test, eval_dict['y_proba'])
    plt.plot(recall_pts, precision_pts, label=f"{name} (PR-AUC = {eval_dict['PR-AUC']:.3f})", color=colors.get(name, '#333333'), lw=2)
plt.xlabel('Recall', fontsize=11)
plt.ylabel('Precision', fontsize=11)
plt.title('Precision-Recall Curves', fontsize=13, fontweight='bold')
plt.legend(loc='lower left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "evaluation", "pr_curves.png"), dpi=200)
plt.close()

# Confusion Matrix for Final Model
plt.figure(figsize=(6, 5))
cm = np.array(test_evaluations[final_model_name]['ConfusionMatrix'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Retained', 'Predicted Churn'],
            yticklabels=['Actual Retained', 'Actual Churn'])
plt.title(f'Confusion Matrix: {final_model_name}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "evaluation", "confusion_matrix.png"), dpi=200)
plt.close()

# Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 5))
comp_melted = comparison_df.melt(id_vars='Model', value_vars=['ROC-AUC', 'F1-Score', 'Recall', 'Precision'], var_name='Metric', value_name='Score')
sns.barplot(x='Model', y='Score', hue='Metric', data=comp_melted, palette='muted', ax=ax)
ax.set_ylim(0, 1.0)
ax.set_title('Comparative Benchmark Across Machine Learning Models', fontsize=13, fontweight='bold')
ax.set_ylabel('Metric Score (0-1)', fontsize=11)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "evaluation", "model_comparison.png"), dpi=200)
plt.close()

# 9. FEATURE IMPORTANCE & SHAP EXPLAINABILITY
print("\n9. GENERATING FEATURE IMPORTANCE & SHAP EXPLAINABILITY")
# Feature Importance Bar Plot (XGBoost gain)
importances = final_model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)

plt.figure(figsize=(9, 7))
sns.barplot(x='Importance', y='Feature', hue='Feature', data=feat_imp_df.head(15), palette='viridis', legend=False)
plt.title('Top 15 Most Important Features (Tuned XGBoost)', fontsize=13, fontweight='bold')
plt.xlabel('Feature Importance (Gain)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "figures", "feature_importance.png"), dpi=200)
plt.close()

# SHAP TreeExplainer
X_test_df = pd.DataFrame(X_test, columns=feature_names)
explainer = shap.TreeExplainer(final_model)
shap_values = explainer(X_test_df)

# SHAP Summary Beeswarm Plot
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values.values, X_test_df, feature_names=feature_names, show=False, max_display=15)
plt.title('SHAP Beeswarm Plot: Impact of Top Features on Churn Probability', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "shap", "shap_summary.png"), dpi=200, bbox_inches='tight')
plt.close()

# SHAP Feature Importance Bar Plot
plt.figure(figsize=(10, 7))
shap.plots.bar(shap_values, max_display=15, show=False)
plt.title('SHAP Mean Absolute Value (|SHAP|): Global Impact', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "shap", "shap_bar.png"), dpi=200, bbox_inches='tight')
plt.close()

# Local Waterfall Plot for High Risk Sample
y_proba_test = test_evaluations[final_model_name]['y_proba']
high_risk_idx = int(np.argmax(y_proba_test))
low_risk_idx = int(np.argmin(y_proba_test))

plt.figure(figsize=(9, 6))
shap.plots.waterfall(shap_values[high_risk_idx], max_display=10, show=False)
plt.title(f'SHAP Waterfall Explanation: High Churn Risk (P = {y_proba_test[high_risk_idx]:.2f})', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "shap", "shap_waterfall_high_risk.png"), dpi=200, bbox_inches='tight')
plt.close()

# Local Waterfall Plot for Low Risk Sample
plt.figure(figsize=(9, 6))
shap.plots.waterfall(shap_values[low_risk_idx], max_display=10, show=False)
plt.title(f'SHAP Waterfall Explanation: Low Churn Risk (P = {y_proba_test[low_risk_idx]:.2f})', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "shap", "shap_waterfall_low_risk.png"), dpi=200, bbox_inches='tight')
plt.close()
print("SHAP figures generated and saved to outputs/shap/")

# 10. SAVE MODEL METADATA
print("\n10. SAVING PRODUCTION MODEL METADATA")
metadata = {
    "model_name": final_model_name,
    "model_class": type(final_model).__name__,
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_version": "1.0.0",
    "dataset": {
        "name": "IBM Telco Customer Churn",
        "provenance": "IBM Developer Sample Datasets",
        "url": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        "total_rows": len(raw_df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "churn_rate_overall": float(cleaned_df['Churn'].mean()),
        "target_variable": "Churn",
        "class_labels": {"0": "No Churn (Retained)", "1": "Churn (Lost Customer)"}
    },
    "feature_schema": {
        "raw_features_count": len(ALL_MODEL_FEATURES),
        "encoded_features_count": len(feature_names),
        "feature_names": feature_names
    },
    "hyperparameters": {k: str(v) for k, v in grid_search.best_params_.items()},
    "default_threshold": 0.50,
    "risk_tiers": {
        "Low": "Probability < 0.30",
        "Medium": "0.30 <= Probability < 0.60",
        "High": "Probability >= 0.60"
    },
    "test_metrics": {
        "ROC-AUC": round(float(test_evaluations[final_model_name]['ROC-AUC']), 4),
        "PR-AUC": round(float(test_evaluations[final_model_name]['PR-AUC']), 4),
        "F1-Score": round(float(test_evaluations[final_model_name]['F1-Score']), 4),
        "Recall": round(float(test_evaluations[final_model_name]['Recall']), 4),
        "Precision": round(float(test_evaluations[final_model_name]['Precision']), 4),
        "Accuracy": round(float(test_evaluations[final_model_name]['Accuracy']), 4),
        "ConfusionMatrix": test_evaluations[final_model_name]['ConfusionMatrix']
    },
    "benchmark_comparison": {
        k: {metric: round(float(v[metric]), 4) for metric in ['ROC-AUC', 'PR-AUC', 'F1-Score', 'Recall', 'Precision', 'Accuracy']}
        for k, v in test_evaluations.items()
    }
}

with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print("Saved models/model_metadata.json")

print("\n" + "="*60)
print("TRAINING AND ARTIFACT GENERATION COMPLETE SUCCESSFULLY!")
print("="*60)
