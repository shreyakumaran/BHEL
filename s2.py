import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Simulate real-time data generation
def generate_real_time_data(normal_interval=1):
    while True:
        # Simulate sensor readings
        if np.random.rand() < 0.9:  # 90% chance of normal operation
            temperature = np.random.normal(75, 5)
            vibration = np.random.normal(5, 1)
            pressure = np.random.normal(30, 2)
            faulty = 0  # Normal operation
        else:
            temperature = np.random.normal(80, 10)  # Higher temperature for fault simulation
            vibration = np.random.normal(8, 2)      # Higher vibration for fault simulation
            pressure = np.random.normal(35, 5)      # Higher pressure for fault simulation
            faulty = 1  # Equipment failure
        
        # Create a data point
        data_point = {
            'temperature': temperature,
            'vibration': vibration,
            'pressure': pressure,
            'faulty': faulty
        }
        
        # Yield the data point
        yield data_point
        
        # Wait for a short period to simulate real-time data streaming
        time.sleep(normal_interval) 

# Data preprocessing
def preprocess_data(data):
    df = pd.DataFrame(data)
    scaler = StandardScaler()
    df[['temperature', 'vibration', 'pressure']] = scaler.fit_transform(df[['temperature', 'vibration', 'pressure']])
    return df

# Feature engineering
def feature_engineering(data):
    data['temp_vibration_ratio'] = data.apply(lambda row: row['temperature'] / row['vibration'] if row['vibration'] != 0 else 0, axis=1)
    data['pressure_temp_product'] = data['pressure'] * data['temperature']
    return data

# Generate simulated historical data
def generate_synthetic_data(n_samples=1000):
    data = {
        'temperature': np.random.normal(75, 5, n_samples),
        'vibration': np.random.normal(5, 1, n_samples),
        'pressure': np.random.normal(30, 2, n_samples),
        'faulty': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    }
    return pd.DataFrame(data)

# Train the model
data = generate_synthetic_data()
data = preprocess_data(data)
data = feature_engineering(data)

X = data[['temperature', 'vibration', 'pressure', 'temp_vibration_ratio', 'pressure_temp_product']]
y = data['faulty']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'predictive_maintenance_model.pkl')

# Load the model
model = joblib.load('predictive_maintenance_model.pkl')

# Monitor simulated real-time data
def monitor_machinery(data, faulty ,model):
    data = preprocess_data([data])
    data = feature_engineering(data)
    prediction = model.predict(data[['temperature', 'vibration', 'pressure', 'temp_vibration_ratio', 'pressure_temp_product']])
    return prediction[0] == faulty
'''def monitor_machinery(data, model):
    data_point = {k: v for k, v in data.items() if k != 'faulty'}  # Remove faulty column
    data = preprocess_data([data_point])
    data = feature_engineering(data)
    prediction = model.predict(data[['temperature', 'vibration', 'pressure', 'temp_vibration_ratio', 'pressure_temp_product']])
    return prediction[0]'''

# Example usage with real-time data simulation
normal_interval = 1 # Seconds
run_duration = 60  # Run the simulation for 1 minute (adjust as needed)

start_time = time.time()
for data_point in generate_real_time_data(normal_interval):
    elapsed_time = time.time() - start_time
    if elapsed_time > run_duration:
        break
    
    prediction = monitor_machinery(data_point, data_point['faulty'], model)
    if prediction == 1:
        print("Alert: Potential failure detected!", data_point)
    else:
        print("All systems normal.", data_point)
    
    time.sleep(normal_interval)

print("Simulation stopped after", run_duration, "seconds.")
