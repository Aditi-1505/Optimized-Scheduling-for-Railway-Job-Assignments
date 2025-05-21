import pandas as pd
import networkx as nx

# # Load the dataset containing train route details
file_path = "/Users/suryapalsinghbisht/Downloads/isl_wise_train_detail_03082015_v1.csv"
df = pd.read_csv(file_path)

# Converting the 'Distance' column to numeric, replacing invalid entries with NaN
df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')

# Initializing a directed graph using NetworkX
G = nx.DiGraph()

# Build the graph by iterating over each train's route
for train_no, group in df.groupby('Train No.'):
    group = group.sort_values(by='islno')
    prev_row = None
    for _, row in group.iterrows():
        if prev_row is not None:
            from_station = prev_row['station Code'].strip()
            to_station = row['station Code'].strip()
            distance = row['Distance'] - prev_row['Distance']
            if pd.notnull(distance) and distance >= 0:
                G.add_edge(from_station, to_station, weight=distance)
        prev_row = row

# Using Dijkstra's algorithm to find the shortest path and its total distance
def find_shortest_path(source, destination):
    try:
        path = nx.dijkstra_path(G, source=source, target=destination, weight='weight')
        distance = nx.dijkstra_path_length(G, source=source, target=destination, weight='weight')
        return path, distance
    except nx.NetworkXNoPath:
        #Handle cases where no path exists between the source and destination
        return None, float('inf')
    except nx.NodeNotFound as e:
        #Handle cases where one or both nodes are not in the graph
        return str(e), None

# Example usage:
source = 'BBS'   # Source station
destination = 'BAM'  # Destination station

# Find the shortest path and total distance using the Dijkstra function
path, total_distance = find_shortest_path(source, destination)

# Display the result
if path:
    print("Shortest path:", " → ".join(path))
    print("Total distance:", total_distance, "km")
else:
    print("No path found or invalid station codes.")