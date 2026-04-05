import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# 1. Load Data from Hugging Face Datasets
# Note: Ensure these paths match your HF Dataset structure exactly
repo_id_data = "SuriyaSR/TourismPackagePurchases"
train_path = f"hf://datasets/{repo_id_data}/train.csv"
test_path = f"hf://datasets/{repo_id_data}/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 2. Define Features and Target
target_col = 'ProdTaken'
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# 3. Identify Feature Types
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# 4. Handle Class Imbalance (Weighting the '1's)
class_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# 5. Preprocessing Pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# 6. Define XGBoost Model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight, 
    random_state=42, 
    use_label_encoder=False, 
    eval_metric='logloss'
)

# 7. Define Hyperparameter Grid (Experimentation)
param_grid = {
    'xgbclassifier__n_estimators': [50, 100, 150],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__subsample': [0.7, 0.8],
}

# 8. Create Pipeline and Run Grid Search
model_pipeline = make_pipeline(preprocessor, xgb_model)
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 9. Evaluate Best Model
best_model = grid_search.best_estimator_
print(f"Best Params found: {grid_search.best_params_}")

y_pred_test = best_model.predict(X_test)
print("\nTest Classification Report:")
print(classification_report(y_test, y_pred_test))

# 10. Model Serialization
model_filename = "tourism_package_model_v1.joblib"
joblib.dump(best_model, model_filename)

# 11. Upload to Hugging Face Model Hub
# Ensure you have set the HF_TOKEN environment variable in your GitHub Secrets or Local Env
hf_token = os.getenv("HF_TOKEN")
api = HfApi(token=hf_token)
model_repo_id = "SuriyaSasiRaja/TourismPackageModel"

try:
    api.repo_info(repo_id=model_repo_id, repo_type="model")
    print(f"Model Repo '{model_repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Creating new model repo: {model_repo_id}")
    create_repo(repo_id=model_repo_id, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=model_filename,
    path_in_repo=model_filename,
    repo_id=model_repo_id,
    repo_type="model",
)

print(f"✅ Model successfully uploaded to Hugging Face: {model_repo_id}")
