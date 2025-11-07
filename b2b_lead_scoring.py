"""
AI-Driven B2B Lead Scoring and Account-Based Marketing (ABM) Optimization System
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("AI-Driven B2B Lead Scoring and ABM Optimization System")
print("=" * 80)
print("\n")

# ============================================================================
# 1. GENERATE SYNTHETIC DATASET
# ============================================================================
print("Step 1: Generating synthetic B2B lead dataset...")

def generate_synthetic_leads(n_samples=2000):
    """Generate synthetic B2B lead data with realistic distributions"""
    
    data = {
        'Company_Size': np.random.choice(['Small', 'Medium', 'Large'], 
                                         n_samples, 
                                         p=[0.5, 0.3, 0.2]),
        'Industry': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Manufacturing', 
                                     'Retail', 'Education', 'Consulting', 'Energy'],
                                    n_samples,
                                    p=[0.25, 0.15, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05]),
        'Annual_Revenue': np.random.lognormal(mean=12, sigma=1.5, size=n_samples),
        'Website_Visits': np.random.poisson(lam=25, size=n_samples),
        'Email_Opens': np.random.poisson(lam=8, size=n_samples),
        'Ad_Clicks': np.random.poisson(lam=5, size=n_samples),
        'Engagement_Score': np.random.beta(a=2, b=5, size=n_samples) * 100,
        'Lead_Source': np.random.choice(['Organic', 'Paid Ads', 'Referral', 
                                        'Social Media', 'Email Campaign'],
                                       n_samples,
                                       p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'Region': np.random.choice(['North America', 'Europe', 'Asia', 'Others'],
                                   n_samples,
                                   p=[0.4, 0.3, 0.2, 0.1]),
        'Previous_Purchases': np.random.poisson(lam=2, size=n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Make Annual_Revenue more realistic (in thousands)
    df['Annual_Revenue'] = df['Annual_Revenue'].round(2)
    
    # Round engagement score to 2 decimal places
    df['Engagement_Score'] = df['Engagement_Score'].round(2)
    
    # Create target variable based on business logic
    # Higher scores for larger companies, more engagement, more revenue
    lead_scores = []
    for idx, row in df.iterrows():
        score = 0
        
        # Company size contribution
        if row['Company_Size'] == 'Large':
            score += 40
        elif row['Company_Size'] == 'Medium':
            score += 20
        
        # Revenue contribution (normalized)
        if row['Annual_Revenue'] > df['Annual_Revenue'].quantile(0.75):
            score += 30
        elif row['Annual_Revenue'] > df['Annual_Revenue'].quantile(0.5):
            score += 15
        
        # Engagement contribution
        if row['Engagement_Score'] > 70:
            score += 20
        elif row['Engagement_Score'] > 50:
            score += 10
        
        # Website visits contribution
        if row['Website_Visits'] > 30:
            score += 10
        
        # Email opens contribution
        if row['Email_Opens'] > 10:
            score += 5
        
        # Previous purchases contribution
        if row['Previous_Purchases'] > 2:
            score += 15
        
        # Add some randomness
        score += np.random.randint(-10, 10)
        
        # Determine lead score class
        if score >= 80:
            lead_scores.append(2)  # High
        elif score >= 50:
            lead_scores.append(1)  # Medium
        else:
            lead_scores.append(0)  # Low
    
    df['Lead_Score'] = lead_scores
    
    return df

# Generate dataset
df = generate_synthetic_leads(2000)
print(f"✓ Generated {len(df)} leads with {len(df.columns)} features")
print(f"✓ Lead Score Distribution: {df['Lead_Score'].value_counts().to_dict()}")
print("\n")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("Step 2: Preprocessing data...")

# Separate features and target
X = df.drop('Lead_Score', axis=1)
y = df['Lead_Score']

# One-hot encode categorical variables
categorical_cols = ['Company_Size', 'Industry', 'Lead_Source', 'Region']
X_encoded = pd.get_dummies(X, columns=categorical_cols, prefix=categorical_cols)

# Normalize/scale numerical features
numerical_cols = ['Annual_Revenue', 'Website_Visits', 'Email_Opens', 
                  'Ad_Clicks', 'Engagement_Score', 'Previous_Purchases']

scaler = StandardScaler()
X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

print(f"✓ One-hot encoded categorical features")
print(f"✓ Scaled numerical features")
print(f"✓ Final feature count: {X_encoded.shape[1]}")
print("\n")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================
print("Step 3: Splitting data (80/20 stratified split)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")
print("\n")

# ============================================================================
# 4. MODEL TRAINING AND EVALUATION
# ============================================================================
print("Step 4: Training and evaluating models...")
print("=" * 80)

models = {
    'Logistic Regression': LogisticRegression(
        multi_class='multinomial', 
        solver='lbfgs', 
        max_iter=1000,
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    ),
    'CatBoost': cb.CatBoostClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=False
    )
}

results = {}
best_model = None
best_f1 = 0
best_model_name = None

# Stratified K-Fold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 80)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # ROC-AUC for multiclass (one-vs-rest)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, 
                               multi_class='ovr', average='weighted')
    except:
        roc_auc = None
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, 
                               cv=cv, scoring='f1_weighted', n_jobs=-1)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Print metrics
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    if roc_auc:
        print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"CV F1-Score (mean ± std): {cv_mean:.4f} ± {cv_std:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Low', 'Medium', 'High']))
    
    # Track best model
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

print("\n" + "=" * 80)
print("\n")

# ============================================================================
# 5. MODEL COMPARISON SUMMARY
# ============================================================================
print("Step 5: Model Comparison Summary")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1_score'] for m in results.keys()],
    'CV F1-Score': [results[m]['cv_mean'] for m in results.keys()],
    'ROC-AUC': [results[m]['roc_auc'] if results[m]['roc_auc'] else 0 
                for m in results.keys()]
})

comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
print(comparison_df.to_string(index=False))
print(f"\n🏆 Best Model: {best_model_name} (F1-Score: {best_f1:.4f})")
print("\n")

# ============================================================================
# 6. SAVE BEST MODEL
# ============================================================================
print("Step 6: Saving best model...")

model_package = {
    'model': best_model,
    'scaler': scaler,
    'feature_names': X_encoded.columns.tolist(),
    'model_name': best_model_name
}

with open('best_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print(f"✓ Saved best model ({best_model_name}) to 'best_model.pkl'")
print("\n")

# ============================================================================
# 7. FEATURE IMPORTANCE VISUALIZATION
# ============================================================================
print("Step 7: Visualizing feature importance...")

# Get feature importance based on model type
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': best_model.feature_importances_
    })
elif hasattr(best_model, 'coef_'):
    # For Logistic Regression, average absolute coefficients across all classes
    coef_abs = np.abs(best_model.coef_)
    feature_importance = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': np.mean(coef_abs, axis=0)
    })
else:
    # Fallback: use permutation importance or equal weights
    feature_importance = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': np.ones(len(X_encoded.columns)) / len(X_encoded.columns)
    })

feature_importance = feature_importance.sort_values('importance', ascending=False)
top_features = feature_importance.head(15)

# Create visualization
plt.figure(figsize=(12, 8))
sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
plt.title(f'Top 15 Feature Importance - {best_model_name}', fontsize=16, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved feature importance plot to 'feature_importance.png'")

# Display top features
print("\nTop 15 Most Important Features:")
print(top_features.to_string(index=False))
print("\n")

print("=" * 80)
print("✅ B2B Lead Scoring System Complete!")
print("=" * 80)

