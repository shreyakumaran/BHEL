import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Simulate data
def generate_synthetic_data(n_samples):
    np.random.seed(42)
    data = {
        'temperature': np.random.normal(75, 5, n_samples),  # normal conditions
        'vibration': np.random.normal(5, 1, n_samples),
        'pressure': np.random.normal(30, 2, n_samples),
        'faulty': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # 10% faulty
    }
    return pd.DataFrame(data)

# Generate dataset
n_samples = 1000
data = generate_synthetic_data(n_samples)

# Split the data
X = data[['temperature', 'vibration', 'pressure']]
y = data['faulty']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
