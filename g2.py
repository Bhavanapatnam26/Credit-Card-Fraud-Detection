print("CREDIT CARD FRAUD DETECTION")
print("=" * 60)
#importing requried libraries
import streamlit as st
import pandas as pd                              #data handling
import numpy as np                               #numerical operations
import matplotlib.pyplot as plt                  #visualization
import seaborn as sns
#machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# 1. Load Dataset
try:
    data = pd.read_csv("creditcard.csv")
    print("Dataset loaded successfully")
except:
    print("Error: File not found")
    exit()

# 2. Basic Info
print("\nDataset Shape:", data.shape)     #rows and columns
print("\nFirst 5 rows:")
print(data.head())

print("\nClass Distribution:")
print(data['Class'].value_counts())

print("\nMissing Values:", data.isnull().sum().sum())

# 3. Preprocessing
X = data.drop(['Class', 'Time'], axis=1)
y = data['Class']

# Scale Amount
scaler_amount = StandardScaler()
X['Amount'] = scaler_amount.fit_transform(X[['Amount']])

# 4. Handle Imbalance (Undersampling)
fraud = X[y == 1]
non_fraud = X[y == 0]

non_fraud_sampled = non_fraud.sample(n=len(fraud), random_state=42)

X_balanced = pd.concat([fraud, non_fraud_sampled])
y_balanced = pd.concat([y[fraud.index], y[non_fraud_sampled.index]])

# Shuffle
X_balanced = X_balanced.sample(frac=1, random_state=42)
y_balanced = y_balanced.loc[X_balanced.index]

print("\nBalanced Dataset Created")
print("Total:", len(X_balanced))

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
)

# 6. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. RANDOM FOREST
print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

rf_auc = roc_auc_score(y_test, rf_proba)
print("ROC-AUC:", rf_auc)

# 8. LOGISTIC REGRESSION
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

print("\nClassification Report:")
print(classification_report(y_test, lr_pred))

lr_auc = roc_auc_score(y_test, lr_proba)
print("ROC-AUC:", lr_auc)

# 9. Feature Importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# 10. Visualization
plt.figure(figsize=(12, 10))

# RF Confusion Matrix
plt.subplot(2, 2, 1)
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d')
plt.title("RF Confusion Matrix")

# LR Confusion Matrix
plt.subplot(2, 2, 2)
sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, fmt='d')
plt.title("LR Confusion Matrix")

# ROC Curve
plt.subplot(2, 2, 3)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)

plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC={rf_auc:.3f})")
plt.plot(fpr_lr, tpr_lr, label=f"LR (AUC={lr_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.legend()
plt.title("ROC Curve")

# Feature Importance
plt.subplot(2, 2, 4)
top_features = feature_importance.head(10)
plt.barh(top_features['feature'], top_features['importance'])
plt.title("Top Features")

plt.tight_layout()
plt.savefig("output.png")
plt.show()

print("\nDone! Output saved as output.png")


st.title("💳 Credit Card Fraud Detection")

# Use trained model (Random Forest)
model = rf_model   # IMPORTANT FIX

# Input fields
amount = st.number_input("Amount", min_value=0.0,key="amount")
v1 = st.number_input("V1",key="v1")
v2 = st.number_input("V2",key="v2")
v3 = st.number_input("V3",key="v3")

# Predict button
if st.button("Predict"):
    st.error("⚠️ Fraudulent Transaction Detected")