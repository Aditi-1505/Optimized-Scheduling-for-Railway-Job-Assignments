import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import heapq
from collections import defaultdict
import networkx as nx

# Load dataset
file_path = "/Users/suryapalsinghbisht/Downloads/isl_wise_train_detail_03082015_v1.csv"
df_full = pd.read_csv(file_path)

# Take first 400 rows to shorten dataset
df = df_full.head(400).copy()

# Clean and convert time columns
df['Arrival time'] = df['Arrival time'].astype(str).str.strip("'")
df['Departure time'] = df['Departure time'].astype(str).str.strip("'")

def to_minutes(t):
    try:
        dt = datetime.strptime(t, "%H:%M:%S")
        return dt.hour * 60 + dt.minute
    except Exception:
        return None

df['arrival_min'] = df['Arrival time'].apply(to_minutes)
df['departure_min'] = df['Departure time'].apply(to_minutes)

# Drop rows with invalid times
df.dropna(subset=['arrival_min', 'departure_min'], inplace=True)
df.loc[df['departure_min'] < df['arrival_min'], 'departure_min'] += 24 * 60

# Build custom graph without networkx
graph = defaultdict(list)

for train_no, group in df.groupby('Train No.'):
    group = group.sort_values('islno')
    prev_row = None
    for _, row in group.iterrows():
        if prev_row is not None:
            from_station = prev_row['station Code'].strip()
            to_station = row['station Code'].strip()
            distance = row['Distance'] - prev_row['Distance']
            if pd.notnull(distance) and distance >= 0:
                graph[from_station].append((to_station, distance))
        prev_row = row

# Dijkstra algorithm
def dijkstra(graph, start, end):
    queue = [(0, start, [start])]
    visited = set()
    
    while queue:
        dist, current, path = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        if current == end:
            return path, dist
        for neighbor, weight in graph.get(current, []):
            if neighbor not in visited:
                heapq.heappush(queue, (dist + weight, neighbor, path + [neighbor]))
    return None, float('inf')

# Visualization 
def visualize_all_simple_paths_with_shortest(source, destination, shortest_path):
    G = nx.DiGraph()
    for node in graph:
        for neighbor, weight in graph[node]:
            G.add_edge(node, neighbor, weight=weight)

    plt.figure(figsize=(14, 12))
    pos = nx.spring_layout(G, seed=42, k=0.7)

    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightgray')
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', arrows=True, alpha=0.3)

    try:
        all_paths = list(nx.all_simple_paths(G, source=source, target=destination))
    except nx.NetworkXNoPath:
        all_paths = []

    all_path_edges = set()
    for path in all_paths:
        for i in range(len(path) - 1):
            all_path_edges.add((path[i], path[i+1]))

    nx.draw_networkx_edges(G, pos, edgelist=list(all_path_edges), edge_color='black', alpha=0.8, width=2)

    if shortest_path and len(shortest_path) > 1:
        shortest_edges = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=shortest_edges, edge_color='red', width=4, arrows=True)
        nx.draw_networkx_nodes(G, pos, nodelist=shortest_path, node_color='skyblue', node_size=400)

    plt.title(f"Paths from {source} to {destination} (black) with Shortest Path (red)", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Platform assignment logic

assignments = []
for station, group in df.groupby('station Code'):
    events = group[['Train No.', 'arrival_min', 'departure_min']].sort_values(by='arrival_min').values
    heap = []  # min-heap of (end_time, platform_id)
    next_platform_id = 1
    platform_map = {}  # maps platform_id to its current end_time

    for train_no, start, end in events:
        if heap and heap[0][0] <= start:
            # Reuse an existing platform
            end_time, platform = heapq.heappop(heap)
        else:
            # Allocate new platform
            platform = next_platform_id
            next_platform_id += 1
        heapq.heappush(heap, (end, platform))
        assignments.append((train_no, station, platform))


result = pd.DataFrame(assignments, columns=['Train No.', 'Station Code', 'Platform'])
print("Platform Assignments Sample:\n", result.head(10))

# Try pathfinding
nodes = list(graph.keys())
print(f"Stations in graph (sample): {nodes[:10]}")

source = 'BAM'
destination = 'LTT'

print(f"Using source: {source} and destination: {destination}")
path, dist = dijkstra(graph, source, destination)

if path:
    print(f"Shortest path from {source} to {destination}: {' → '.join(path)}")
    print(f"Total shortest distance: {dist} km")
    visualize_all_simple_paths_with_shortest(source, destination, path)
else:
    print(f"No path found from {source} to {destination}.")
