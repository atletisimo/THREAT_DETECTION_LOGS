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


#КРЕИРАЊЕ НА ГРАФ

G = nx.DiGraph()
for index, row in df.iterrows():
    G.add_edge(row['source_ip'], row['dest_ip'])

#Пресметување на graph features

print("Calculating graph features...")

in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())
degree = dict(G.degree())
pagerank_scores = pagerank(G)

print("Creating graph features DataFrame...")

# Create a DataFrame from the calculated features
graph_features_df = pd.DataFrame({
    'in_degree': pd.Series(in_degree),
    'out_degree': pd.Series(out_degree),
    'degree': pd.Series(degree),
    'pagerank': pd.Series(pagerank_scores)
})

# Ensure the index is named for clarity
graph_features_df.index.name = 'ip_address'

print("Graph features DataFrame created successfully.")
print("First 5 rows of graph_features_df:\n", graph_features_df.head())

print("Merging graph features into main DataFrame...")

# Merge source IP features
df = df.merge(graph_features_df, left_on='source_ip', right_index=True, how='left',
              suffixes=('_source', '_dest'))

# Merge destination IP features
df = df.merge(graph_features_df, left_on='dest_ip', right_index=True, how='left',
              suffixes=('_source', '_dest_ip_features'))

# Rename columns to reflect their origin (source or destination)
df.rename(columns={
    'in_degree_source': 'source_in_degree',
    'out_degree_source': 'source_out_degree',
    'degree_source': 'source_degree',
    'pagerank_source': 'source_pagerank',
    'in_degree_dest_ip_features': 'dest_in_degree',
    'out_degree_dest_ip_features': 'dest_out_degree',
    'degree_dest_ip_features': 'dest_degree',
    'pagerank_dest_ip_features': 'dest_pagerank'
}, inplace=True)

print("Graph features merged successfully.")
print("First 5 rows of df with new features:\n", df.head())


#ПОМОШНИ ФУНКЦИИ

def calculate_metrics(y_true, y_pred):
    """Calculate ACC, FAR, and UND"""
    cm = confusion_matrix(y_true, y_pred)

    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    ACC = ((TP + TN) / (TP + TN + FP + FN)) * 100

    FAR = (FP / (FP + TN)) * 100 if (FP + TN) > 0 else 0

    UND = (FN / (FN + TP)) * 100 if (FN + TP) > 0 else 0

    return {
        'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
        'Accuracy': ACC, 'FAR': FAR, 'UND': UND
    }


def print_metrics(model_name, metrics):
    print(f"\n{model_name} Results:")
    print("-" * 50)
    print(f"Confusion Matrix:")
    print(f"  TN: {metrics['TN']:6d}  |  FP: {metrics['FP']:6d}")
    print(f"  FN: {metrics['FN']:6d}  |  TP: {metrics['TP']:6d}")
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy (ACC):        {metrics['Accuracy']:.2f}%")
    print(f"  False Alarm Rate (FAR): {metrics['FAR']:.2f}%")
    print(f"  Un-Detection Rate (UND): {metrics['UND']:.2f}%")


results = {}

#Подготовка на data-та за ML
print("Preparing data for ML...")

# Drop the 'timestamp' and 'threat_label' columns to create the feature set X
X = df.drop(columns=['timestamp', 'threat_label'])

# Create a Series y containing only the 'threat_label' column
y = df['threat_label']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data prepared and split successfully.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

#StandardScaler
print("Scaling features using StandardScaler...")

# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled successfully.")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")




#ЛОГИСТИЧКА РЕГРЕСИЈА

print("\nTraining Logistic Regression model...")

# Initialize Logistic Regression model
log_reg_model = LogisticRegression(random_state=42, solver='liblinear', multi_class='auto')

# Train the model
log_reg_model.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred_log_reg = log_reg_model.predict(X_test_scaled)

# Calculate metrics
metrics_log_reg = calculate_metrics(y_test, y_pred_log_reg)
results['Logistic Regression'] = metrics_log_reg

# Print metrics
print_metrics('Logistic Regression', metrics_log_reg)

print("Logistic Regression model trained and evaluated.")



#RANDOM FOREST

print("\nTraining Random Forest Classifier model...")

# Initialize Random Forest Classifier model
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred_rf = rf_model.predict(X_test_scaled)

# Calculate metrics
metrics_rf = calculate_metrics(y_test, y_pred_rf)
results['Random Forest'] = metrics_rf

# Print metrics
print_metrics('Random Forest', metrics_rf)

print("Random Forest Classifier model trained and evaluated.")







#NAIVE BAYES

print("\nTraining Gaussian Naive Bayes model...")

# Initialize Gaussian Naive Bayes model
gnb_model = GaussianNB()

# Train the model
gnb_model.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred_gnb = gnb_model.predict(X_test_scaled)

# Calculate metrics
metrics_gnb = calculate_metrics(y_test, y_pred_gnb)
results['Gaussian Naive Bayes'] = metrics_gnb

# Print metrics
print_metrics('Gaussian Naive Bayes', metrics_gnb)

print("Gaussian Naive Bayes model trained and evaluated.")






#ДРВО НА ОДЛУКА 

print("\nTraining Decision Tree Classifier model...")

# Initialize Decision Tree Classifier model
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
dt_model.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred_dt = dt_model.predict(X_test_scaled)

# Calculate metrics
metrics_dt = calculate_metrics(y_test, y_pred_dt)
results['Decision Tree'] = metrics_dt

# Print metrics
print_metrics('Decision Tree', metrics_dt)

print("Decision Tree Classifier model trained and evaluated.")


#СПОРЕДБА

print("Creating summary DataFrame...")

# Initialize an empty list to store the model performance data
model_performance_data = []

# Iterate through the results dictionary
for model_name, metrics in results.items():
    # Create a dictionary containing 'Model', 'Accuracy (%)', 'FAR (%)', and 'UND (%)'
    model_performance_data.append({
        'Model': model_name,
        'Accuracy (%)': metrics['Accuracy'],
        'FAR (%)': metrics['FAR'],
        'UND (%)': metrics['UND']
    })

# Create a pandas DataFrame from the list of dictionaries
metrics_df = pd.DataFrame(model_performance_data)

print("Summary DataFrame created successfully.")
print("Model performance summary:\n", metrics_df)
print("Model Performance Summary:")
print(metrics_df)




#КОМПАРАТИВНА ВИЗУЕЛИЗАЦИЈА

import matplotlib.pyplot as plt
import seaborn as sns

print("Visualizing comparative model performance...")

# Melt the DataFrame to transform it into a long format suitable for plotting
metrics_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Value')

# Create a bar plot using seaborn.catplot
plt.figure(figsize=(12, 7))
sns.catplot(x='Metric', y='Value', hue='Model', data=metrics_melted, kind='bar', height=6, aspect=1.5)

# Set the title and labels
plt.title('Comparative Model Performance', fontsize=16)
plt.xlabel('Metric', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.ylim(0, 100)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("Comparative model performance visualization generated.")








#ДЕТЕКЦИЈА НА АНОМАЛИИ

print("Applying Isolation Forest for anomaly detection...")

# 1. Select all columns from the DataFrame df except 'timestamp' and 'threat_label'
#    to create the feature set for anomaly detection.
X_anomaly = df.drop(columns=['timestamp', 'threat_label'])

# 2. Initialize the IsolationForest model with contamination=0.01 and random_state=42.
isolation_forest = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)

# 3. Fit the IsolationForest model to the X_anomaly dataset.
isolation_forest.fit(X_anomaly)

# 4. Predict the anomaly labels for each data point using the fitted model
#    and store them in a new column named anomaly_label directly in the main DataFrame df.
df['anomaly_label'] = isolation_forest.predict(X_anomaly)

print("Isolation Forest applied successfully.")
print("First 5 rows of df with new 'anomaly_label':\n", df.head())
print("\nNumber of anomalies detected (-1):", df['anomaly_label'].value_counts()[-1])
print("\nDistribution of anomaly_label:")
print(df['anomaly_label'].value_counts())

print("\nCross-tabulation of anomaly_label and threat_label:")
cross_tab = pd.crosstab(df['anomaly_label'], df['threat_label'])
print(cross_tab)
































#SCALING НА АНОМАЛИИ

print("Scaling X_anomaly using the previously fitted scaler...")
X_anomaly_scaled = scaler.transform(X_anomaly)
print("X_anomaly scaled successfully.")

print("Predicting threat labels using the Decision Tree model...")
df['dt_predicted_threat'] = dt_model.predict(X_anomaly_scaled)
print("Decision Tree predictions added to df.")

print("Identifying risky events...")
risky_events_df = df[(df['anomaly_label'] == -1) | (df['dt_predicted_threat'] != 0)]
print(f"Number of risky events identified: {len(risky_events_df)}")

print("Extracting unique risky IP addresses...")
risky_source_ips = set(risky_events_df['source_ip'].unique())
risky_dest_ips = set(risky_events_df['dest_ip'].unique())
all_risky_ips = risky_source_ips.union(risky_dest_ips)
print(f"Total unique risky IP addresses: {len(all_risky_ips)}")

print("Creating risky IPs DataFrame and merging with graph features...")
risky_ips_df = pd.DataFrame(list(all_risky_ips), columns=['ip_address'])
risky_ips_df = risky_ips_df.set_index('ip_address')

# Merge with graph_features_df
risky_ips_with_features = risky_ips_df.merge(graph_features_df, left_index=True, right_index=True, how='left')
print("Risky IPs merged with graph features.")

print("Calculating total risky events per IP...")
# Calculate total risky events for each IP
risky_events_count = pd.Series(index=all_risky_ips, dtype=int)
for ip in all_risky_ips:
    count = risky_events_df[(risky_events_df['source_ip'] == ip) | (risky_events_df['dest_ip'] == ip)].shape[0]
    risky_events_count.loc[ip] = count

risky_ips_with_features['total_risky_events'] = risky_events_count

print("Sorting by total risky events and PageRank...")
final_risky_ips = risky_ips_with_features.sort_values(by=['total_risky_events', 'pagerank'], ascending=[False, False])

print("Top 10 Risky IP Addresses:")
print(final_risky_ips.head(10))

#ВИЗУЕЛИЗАЦИЈА 

import matplotlib.pyplot as plt
import seaborn as sns

print("Visualizing top 10 risky IP addresses...")

# Select the top 10 risky IP addresses
top_10_risky_ips = final_risky_ips.head(10)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Plot total_risky_events
sns.barplot(x=top_10_risky_ips.index.astype(str), y='total_risky_events', data=top_10_risky_ips, ax=axes[0], palette='viridis')
axes[0].set_title('Top 10 Risky IPs by Total Risky Events', fontsize=14)
axes[0].set_xlabel('IP Address', fontsize=12)
axes[0].set_ylabel('Total Risky Events', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

# Plot pagerank
sns.barplot(x=top_10_risky_ips.index.astype(str), y='pagerank', data=top_10_risky_ips, ax=axes[1], palette='magma')
axes[1].set_title('Top 10 Risky IPs by PageRank Score', fontsize=14)
axes[1].set_xlabel('IP Address', fontsize=12)
axes[1].set_ylabel('PageRank Score', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("Top 10 risky IP addresses visualization generated.")





#LABEL PROPAGATION 
print("Preparing data for Label Propagation...")

# 1. Calculate the total number of samples in y_train
total_samples = len(y_train)
print(f"Total samples in y_train: {total_samples}")

# 2. Calculate the number of 'normal' samples (where y_train is 0) and 'attack' samples (where y_train is 1 or 2)
normal_samples = (y_train == 0).sum()
attack_samples = ((y_train == 1) | (y_train == 2)).sum()
print(f"Normal samples (threat_label 0): {normal_samples}")
print(f"Attack samples (threat_label 1 or 2): {attack_samples}")

# 3. Create a copy of y_train and store it in a new NumPy array called y_labeled
y_labeled = np.copy(y_train)

# 4. Determine the number of samples that need to be masked (99% of the total samples)
num_to_mask = int(0.99 * total_samples)
print(f"Number of samples to mask (99%): {num_to_mask}")

# 5. Generate random indices for the samples to be masked using np.random.choice
# Ensure that at least one sample from each class is retained, if possible.
# For simplicity, we are randomly masking 99% of all samples, which might mask all of a minority class.
# A more robust approach would be to mask per class.
masked_indices = np.random.choice(total_samples, size=num_to_mask, replace=False)

# 6. Set the values at these random indices in y_labeled to -1, which signifies an unlabeled sample.
y_labeled[masked_indices] = -1

# 7. Print the value counts of the newly created y_labeled array to confirm the masking.
print("\nValue counts of y_labeled after masking:")
print(pd.Series(y_labeled).value_counts())

print("Data prepared for Label Propagation successfully.")








































































