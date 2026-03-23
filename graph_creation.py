import pandas as pd
import networkx as nx

# Читање на dataset
df = pd.read_csv("cyber_logs.csv", parse_dates=['timestamp'])

# Проверка на првите редови
print(df.head())

#  Креирање на директен граф (Directed Graph)
G = nx.DiGraph()

#  Додавање на nodes и edges
for _, row in df.iterrows():
    src = row['source_ip']
    dst = row['dest_ip']
    
    # Додавање на node-ови
    G.add_node(src)
    G.add_node(dst)
    
    # Додавање на edge со атрибути
    G.add_edge(
        src, dst,
        protocol=row['protocol'],
        action=row['action'],
        bytes_transferred=row['bytes_transferred'],
        log_type=row['log_type'],
        threat_label=row['threat_label']
    )

print(f"Графот има {G.number_of_nodes()} nodes и {G.number_of_edges()} edges")
