# ===============================
# 0. IMPORTS
# ===============================
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelPropagation

# GNN imports
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# ===============================
# 1. LOAD & PREPROCESS DATA
# ===============================
df = pd.read_csv("cyber_logs.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract time features
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month

# Encode categorical features
le = LabelEncoder()
for col in ['protocol', 'action', 'log_type']:
    df[col] = le.fit_transform(df[col])

# Encode target
df['label'] = df['threat_label'].map({'benign': 0, 'suspicious': 1})

# ===============================
# 2. CREATE GRAPH
# ===============================
G = nx.from_pandas_edgelist(df, source='source_ip', target='dest_ip', create_using=nx.DiGraph())

# Add graph-based features
pagerank = nx.pagerank(G)
betweenness = nx.betweenness_centrality(G)
clustering = nx.clustering(G.to_undirected())

# Map IP to graph features
df['src_degree'] = df['source_ip'].apply(lambda x: G.degree(x))
df['dst_degree'] = df['dest_ip'].apply(lambda x: G.degree(x))
df['pagerank'] = df['source_ip'].map(pagerank)
df['betweenness'] = df['source_ip'].map(betweenness)
df['clustering'] = df['source_ip'].map(clustering)

# Final features
features = [
    'protocol', 'action', 'log_type', 'bytes_transferred',
    'hour', 'day', 'month',
    'src_degree', 'dst_degree',
    'pagerank', 'betweenness', 'clustering'
]
X = df[features].fillna(0)
y = df['label']

# ===============================
# 3. CLASSICAL ML
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ml_models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

print("=== Classical ML Results ===")
for name, model in ml_models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"{name} Accuracy: {acc:.4f}")

# ===============================
# 4. LABEL PROPAGATION (Semi-Supervised)
# ===============================
labels_lp = np.array(y)
train_idx_lp, test_idx_lp = train_test_split(range(len(labels_lp)), test_size=0.2, random_state=42)
labels_lp[test_idx_lp] = -1  # unknown for test nodes

lp_model = LabelPropagation()
lp_model.fit(X, labels_lp)
pred_lp = lp_model.transduction_

test_acc_lp = (pred_lp[test_idx_lp] == np.array(y)[test_idx_lp]).mean()
print(f"\nLabel Propagation Accuracy (test nodes): {test_acc_lp:.4f}")

# ===============================
# 5. GRAPH NEURAL NETWORK (GNN)
# ===============================
# Map IP to index
ip_map = {ip: i for i, ip in enumerate(G.nodes())}

edges = [[ip_map[u], ip_map[v]] for u, v in G.edges()]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
x = torch.tensor(X.values, dtype=torch.float)
y_torch = torch.tensor(y.values, dtype=torch.long)

num_nodes = x.shape[0]
train_idx_gnn, test_idx_gnn = train_test_split(range(num_nodes), test_size=0.2, random_state=42)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx_gnn] = True
test_mask[test_idx_gnn] = True

data = Data(x=x, edge_index=edge_index, y=y_torch, train_mask=train_mask, test_mask=test_mask)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN(in_channels=x.shape[1], hidden_channels=16, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
out = model(data)
pred_gnn = out.argmax(dim=1)
correct = (pred_gnn[data.test_mask] == data.y[data.test_mask]).sum()
acc_gnn = int(correct) / int(data.test_mask.sum())
print(f"\nGNN Test Accuracy: {acc_gnn:.4f}")
