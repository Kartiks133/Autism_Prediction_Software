# autism_model.py
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib

# ---------- SETTINGS ----------
CSV_PATH = "autism_data.csv"  # Your dataset
RANDOM_STATE = 42
# ------------------------------

# 0️⃣ Load dataset & select important columns
selected_columns = [
    "age", "gender", "jundice", "used_app_before",
    "A1_Score","A2_Score","A3_Score","A4_Score","A5_Score",
    "A6_Score","A7_Score","A8_Score","A9_Score","A10_Score",
    "result"  # target column
]

df = pd.read_csv(CSV_PATH)
df = df[selected_columns]

# 1️⃣ Prepare features and target
X = df.drop(columns=["result"])
y = df["result"]

# 2️⃣ Ensure target is strictly 0/1
if y.dtype != int:
    y = y.astype(str).str.strip().str.lower().map({'yes':1,'no':0}).fillna(0).astype(int)

print("Unique values in target:", y.unique())

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# 4️⃣ Preprocessing pipelines
numeric_selector = make_column_selector(dtype_include=np.number)
categorical_selector = make_column_selector(dtype_include=object)

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_selector),
    ('cat', categorical_pipeline, categorical_selector)
])

# 5️⃣ Handle class imbalance with SMOTE
minority_ratio = (y_train == 1).mean()
if minority_ratio < 0.3:
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('clf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
    ])
else:
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
    ])

# 6️⃣ Hyperparameter tuning
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 20]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best parameters:", grid.best_params_)

# 7️⃣ Evaluate
y_pred = best_model.predict(X_test)
try:
    y_proba = best_model.predict_proba(X_test)[:,1]
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
except ValueError:
    print("ROC AUC cannot be computed: target is not strictly binary.")

print("Classification Report:\n", classification_report(y_test, y_pred))

# 8️⃣ Save model
joblib.dump(best_model, "autism_model.joblib")
print("Model saved as autism_model.joblib")
