import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load training data using absolute path
X_train = pd.read_csv(os.path.join(BASE_DIR, "data", "X_train.csv"))
y_train = pd.read_csv(os.path.join(BASE_DIR, "data", "y_train.csv"))

# Define column names
text_column = "review_text"
category_column = "category"
numeric_columns = ["price"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(), text_column),
        ("category", OneHotEncoder(handle_unknown="ignore"), [category_column]),
        ("numeric", "passthrough", numeric_columns)
    ]
)

# Create pipeline
model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# Train model
model.fit(X_train, y_train.values.ravel())

# Save model
model_path = os.path.join(BASE_DIR, "model", "model_v1.pkl")
joblib.dump(model, model_path)

print("✅ Model trained successfully and saved as model_v1.pkl")
