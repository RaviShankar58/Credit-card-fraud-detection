import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# =========================
# Step 1: Load and Preprocess Dataset
# =========================
df = pd.read_csv('creditcard.csv')

# Drop 'Time', scale 'Amount'
df = df.drop(['Time'], axis=1)
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# =========================
# Step 2: Train-Test Split
# =========================
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# =========================
# Step 3: XGBoost (Before SMOTE)
# =========================
print("\n=========== XGBoost Model (Before SMOTE) ===========\n")

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

print("Confusion Matrix (Before SMOTE):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report (Before SMOTE):")
print(classification_report(y_test, y_pred))

# =========================
# Step 4: Apply SMOTE
# =========================
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# =========================
# Step 5: XGBoost (After SMOTE)
# =========================
print("\n=========== XGBoost Model (After SMOTE) ===========\n")

xgb_smote = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_smote.fit(X_train_res, y_train_res)
y_pred_smote = xgb_smote.predict(X_test)

print("Confusion Matrix (After SMOTE):")
print(confusion_matrix(y_test, y_pred_smote))
print("\nClassification Report (After SMOTE):")
print(classification_report(y_test, y_pred_smote))
