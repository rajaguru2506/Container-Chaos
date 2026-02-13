import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "model", "model_v1.pkl"))

sample = pd.DataFrame({
    "review_text": ["Amazing product"],
    "category": ["Electronics"],
    "price": [1200]
})

prediction = model.predict(sample)

print("Predicted Rating:", prediction)
