import kagglehub
path = kagglehub.dataset_download("aryan208/cybersecurity-threat-detection-logs")

#ИМПОРТИРАЊЕ НА ПОТРЕБНИТЕ БИБЛИОТЕКИ

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import confusion_matrix
import networkx as nx
from networkx import pagerank
import warnings
warnings.filterwarnings('ignore')

#DATA LOADING AND PREPROCESSING


# ===============================
# LOADING AND PREPROCESSING DATA
# ===============================
print("="*80)
print("LOADING AND PREPROCESSING DATA")
print("="*80)

df = pd.read_csv('/root/.cache/kagglehub/datasets/aryan208/cybersecurity-threat-detection-logs/versions/1/cybersecurity_threat_detection_logs.csv')
print("Број на редици:", df.shape[0])
print("\nБрој на колони:", df.shape[1])
print("\nПрви 5 редици:\n", df.head())
print("\nПодатоци за dataset-от:\n")
print(df.info())
#cleaning
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
print(df["timestamp"])
print("Проверка за missing values:",(df.isnull().sum()))

print("Threat label:",df["threat_label"].dtype)
print(df["threat_label"].unique())



print("\nUnique values before encoding:")
for column in df.select_dtypes(include=['object']).columns:
    print(f"{column}: {df[column].unique()}")

# Initialize LabelEncoder
le = LabelEncoder()

# Encode categorical features
for column in ['source_ip', 'dest_ip', 'protocol', 'action', 'log_type', 'user_agent', 'request_path']:
    df[column] = le.fit_transform(df[column])

# Encode the target variable 'threat_label'
df['threat_label'] = le.fit_transform(df['threat_label'])

print("\nUnique values after encoding:")
for column in ['source_ip', 'dest_ip', 'protocol', 'action', 'log_type', 'user_agent', 'request_path', 'threat_label']:
    print(f"{column}: {df[column].unique()}")

