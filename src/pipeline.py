"""
Pipeline and Preprocessing Module for Customer Churn Prediction.
Guarantees 100% training-serving parity across Notebook and Streamlit.
"""

from typing import List, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Raw dataset definitions
RAW_FEATURE_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# Numeric and Categorical feature lists for modeling
NUMERIC_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'total_services',
    'monthly_to_total_ratio', 'avg_monthly_charges_diff'
]

CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group',
    'has_tech_support_or_security', 'is_long_term_contract', 'automatic_payment'
]

ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def clean_raw_data(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Clean raw Telco customer records.
    - Resolves blank spaces in TotalCharges for tenure=0 records
    - Coerces numerical data types
    - Preserves or drops customerID
    - Encodes Churn target if training
    """
    data = df.copy()

    # TotalCharges contains whitespace for new customers (tenure=0)
    if 'TotalCharges' in data.columns:
        data['TotalCharges'] = pd.to_numeric(
            data['TotalCharges'].astype(str).str.strip(), errors='coerce'
        ).fillna(0.0)

    # Ensure numeric columns are properly cast
    if 'tenure' in data.columns:
        data['tenure'] = pd.to_numeric(data['tenure'], errors='coerce').fillna(0).astype(int)
    if 'MonthlyCharges' in data.columns:
        data['MonthlyCharges'] = pd.to_numeric(data['MonthlyCharges'], errors='coerce').fillna(0.0)
    if 'SeniorCitizen' in data.columns:
        data['SeniorCitizen'] = pd.to_numeric(data['SeniorCitizen'], errors='coerce').fillna(0).astype(int)

    # Clean Churn target if present
    if is_training and 'Churn' in data.columns:
        data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})

    return data


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate domain-specific features related to customer churn:
    1. tenure_group: Lifecycle phases ('0-12m', '13-24m', '25-48m', '49-60m', '>60m')
    2. total_services: Count of active subscribed telecom/internet services
    3. has_tech_support_or_security: Protective security flags
    4. is_long_term_contract: 1/2 year contract commitment indicator
    5. automatic_payment: Autopay indicator (bank transfer / credit card)
    6. monthly_to_total_ratio: Ratio of current month spend to total accumulated spend
    7. avg_monthly_charges_diff: Current monthly bill vs average historical monthly rate
    """
    data = df.copy()

    # 1. Tenure Groups
    bins = [-1, 12, 24, 48, 60, 120]
    labels = ['0-12m', '13-24m', '25-48m', '49-60m', '>60m']
    data['tenure_group'] = pd.cut(data['tenure'], bins=bins, labels=labels).astype(str)

    # 2. Total Services Subscribed
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    service_sum = pd.Series(0, index=data.index)
    for col in service_cols:
        if col in data.columns:
            service_sum += (data[col] == 'Yes').astype(int)
    data['total_services'] = service_sum

    # 3. Protective Services Indicator
    has_sec = (data['OnlineSecurity'] == 'Yes') if 'OnlineSecurity' in data.columns else False
    has_tech = (data['TechSupport'] == 'Yes') if 'TechSupport' in data.columns else False
    data['has_tech_support_or_security'] = (has_sec | has_tech).astype(int)

    # 4. Long-Term Contract Flag
    if 'Contract' in data.columns:
        data['is_long_term_contract'] = data['Contract'].isin(['One year', 'Two year']).astype(int)
    else:
        data['is_long_term_contract'] = 0

    # 5. Automatic Payment Method Flag
    if 'PaymentMethod' in data.columns:
        data['automatic_payment'] = data['PaymentMethod'].str.contains('automatic', case=False, na=False).astype(int)
    else:
        data['automatic_payment'] = 0

    # 6. Monthly to Total Charges Ratio
    data['monthly_to_total_ratio'] = data['MonthlyCharges'] / (data['TotalCharges'] + 1.0)

    # 7. Price Shock / Creep (Current vs Historical average)
    avg_hist = data['TotalCharges'] / (data['tenure'] + 1e-5)
    data['avg_monthly_charges_diff'] = data['MonthlyCharges'] - avg_hist

    return data


def build_preprocessor() -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer for numerical scaling and categorical one-hot encoding.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERIC_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor


def extract_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extract clean, human-readable transformed feature names from fitted ColumnTransformer.
    """
    raw_names = preprocessor.get_feature_names_out()
    clean_names = [name.replace('num__', '').replace('cat__', '') for name in raw_names]
    return clean_names


def prepare_inference_df(
    raw_df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Execute full inference preprocessing pipeline:
    raw dataframe -> clean -> feature engineer -> transform -> named DataFrame.
    """
    cleaned = clean_raw_data(raw_df, is_training=False)
    engineered = engineer_features(cleaned)
    X_transformed = preprocessor.transform(engineered[ALL_MODEL_FEATURES])
    return pd.DataFrame(X_transformed, columns=feature_names, index=raw_df.index)


def get_risk_tier(probability: float) -> Tuple[str, str, str]:
    """
    Classify probability into risk tiers with color codes.
    Returns: (risk_level, css_color, badge_label)
    """
    if probability < 0.30:
        return "Low", "#28a745", "Low Churn Risk"
    elif probability < 0.60:
        return "Medium", "#ffc107", "Medium Churn Risk"
    else:
        return "High", "#dc3545", "High Churn Risk"


def generate_retention_recommendations(customer: Union[pd.Series, Dict[str, Any]], probability: float) -> List[Dict[str, str]]:
    """
    Generate actionable business retention strategies tailored to customer attributes.
    """
    recs = []

    # Contract recommendation
    contract = customer.get('Contract', '')
    if contract == 'Month-to-month':
        recs.append({
            'strategy': 'Contract Conversion Incentive',
            'action': 'Offer a 15% discount or billing credit for committing to an Annual (1-Year) subscription.',
            'impact': 'High (Long-term contracts reduce churn risk by up to 70%)'
        })

    # Tech Support / Online Security
    tech_support = customer.get('TechSupport', '')
    online_sec = customer.get('OnlineSecurity', '')
    if tech_support == 'No' or online_sec == 'No':
        recs.append({
            'strategy': 'Protective Services Bundle',
            'action': 'Provide 3-6 months free Tech Support and Online Security to boost service engagement.',
            'impact': 'Medium-High (Increases switching barriers and resolves customer technical frustrations)'
        })

    # Payment Method
    payment = customer.get('PaymentMethod', '')
    if 'check' in payment.lower():
        recs.append({
            'strategy': 'Automated Billing Enrollment',
            'action': 'Incentivize enrollment in Credit Card or Bank Transfer autopay with a one-time $10 account credit.',
            'impact': 'Medium (Reduces involuntary churn and manual payment friction)'
        })

    # High Monthly Charges
    monthly = float(customer.get('MonthlyCharges', 0))
    if monthly > 80:
        recs.append({
            'strategy': 'Value Optimization Review',
            'action': 'Reach out for an account audit to optimize service tiers or bundle streaming services at a discounted rate.',
            'impact': 'High (Addresses price sensitivity and perceived cost-to-value gap)'
        })

    # Short Tenure
    tenure = int(customer.get('tenure', 0))
    if tenure <= 6:
        recs.append({
            'strategy': 'Proactive Onboarding Follow-Up',
            'action': 'Schedule a dedicated customer success check-in call to ensure satisfaction and product adoption.',
            'impact': 'High (The first 6 months exhibit the highest customer attrition rates)'
        })

    # Fallback if no specific triggers hit
    if not recs:
        recs.append({
            'strategy': 'Loyalty Reward & Engagement',
            'action': 'Provide personalized loyalty reward points and survey customer satisfaction.',
            'impact': 'Medium (Maintains healthy NPS and positive brand connection)'
        })

    return recs
