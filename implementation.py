# =====================================
# STEP 1: Load and Analyze the Data
# =====================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("\n=========== STEP 1: Load and Explore Dataset ===========\n")

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Show the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Count of each class
print("\nClass distribution:")
print(df['Class'].value_counts())

# Plot class distribution
sns.countplot(data=df, x='Class')
plt.title("Class Distribution (0 = Genuine, 1 = Fraud)")
plt.show()


# =====================================
# STEP 2: Preprocessing and Scaling
# =====================================

from sklearn.preprocessing import StandardScaler

print("\n=========== STEP 2: Preprocessing and Scaling ===========\n")

# Drop 'Time' column
df = df.drop(['Time'], axis=1)

# Scale 'Amount' and replace it
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Optional: Check if null values exist
print("Any null values in each column?")
print(df.isnull().sum())

# Optional: Show updated columns
print("\nUpdated column names after preprocessing:")
print(df.columns)


# =====================================
# STEP 3: Train-Test Split
# =====================================

from sklearn.model_selection import train_test_split

print("\n=========== STEP 3: Train-Test Split ===========\n")

# Separate features and target
X = df.drop('Class', axis=1)  # All columns except 'Class'
y = df['Class']               # Target column

# Train-test split (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,      # keeps the fraud/genuine ratio same in both sets
    random_state=42  # for reproducibility
)

# Check split sizes
print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
print("Frauds in train set:", sum(y_train))
print("Frauds in test set:", sum(y_test))


# =====================================
# STEP 4: Train and Evaluate Models
# =====================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("\n=========== STEP 4: Model Training and Evaluation ===========\n")

# Initialize models
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)

# Train both models
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predict on test set
dt_preds = dt.predict(X_test)
rf_preds = rf.predict(X_test)

# Evaluate each model
print("\n=== Decision Tree Metrics ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_preds))
print("Classification Report:")
print(classification_report(y_test, dt_preds))

print("\n=== Random Forest Metrics ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))
print("Classification Report:")
print(classification_report(y_test, rf_preds))

# =====================================
#  STEP 5: Handle Imbalance with SMOTE
# =====================================

from imblearn.over_sampling import SMOTE

print("\n=========== STEP 5: Apply SMOTE (Balance Classes) ===========\n")

# Apply SMOTE to training data
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Check new class distribution
print("Class distribution AFTER applying SMOTE:")
print(pd.Series(y_train_res).value_counts())

# Retrain Random Forest on resampled data
rf_smote = RandomForestClassifier(n_estimators=10,random_state=42)
rf_smote.fit(X_train_res, y_train_res)

# Predict on original test set
y_pred_smote = rf_smote.predict(X_test)

# Evaluate
print("\n=== Random Forest Metrics After SMOTE ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_smote))
print("Classification Report:")
print(classification_report(y_test, y_pred_smote))
