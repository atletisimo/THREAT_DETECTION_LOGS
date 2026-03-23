#### Data Loading & Preprocessing

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("synthetic_network_logs.csv", parse_dates=['timestamp'])

# Feature Engineering
df['hour'] = df['timestamp'].dt.hour

# Encode categorical features
df['protocol'] = LabelEncoder().fit_transform(df['protocol'])
df['log_type'] = LabelEncoder().fit_transform(df['log_type'])
df['action'] = LabelEncoder().fit_transform(df['action'])

# Features and target
X = df[['protocol', 'log_type', 'action', 'bytes_transferred', 'hour']]
y = df['threat_label']

# Encode target
y = LabelEncoder().fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_train[['bytes_transferred','hour']] = scaler.fit_transform(X_train[['bytes_transferred','hour']])
X_test[['bytes_transferred','hour']] = scaler.transform(X_test[['bytes_transferred','hour']])
