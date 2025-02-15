import torch
import networkx as nx
from torch_geometric.data import Data
from mydatasets.mydataloaders import get_graph_dataloaders
from utils.construct_graph import get_graph
import pandas as pd


def graph_stats(data, edges_dic_num, conv_indices_to_keep, my_new_mask_idx):
    assert len(edges_dic_num.keys()) <= 1, "length of edges dic num is greater than 1"
    edge_list = []
    for k in edges_dic_num.keys():
        edge_list = edges_dic_num[k]
        edge_list = sorted(edge_list, key=lambda x: (x[0], x[1]))
        edge_list = torch.tensor(edge_list)

    texts = []
    features_kept = []
    for i in conv_indices_to_keep:
        texts.append(data.x_text[i][0]['body'])
        features_kept.append(data.x[i])
    if conv_indices_to_keep is None:
        for i in len(data.x_text):
            texts.append(data.x_text[i][0]['body'])
            features_kept.append(data.x[i])

    graph = Data(x=features_kept, edge_index=edge_list.t().contiguous())        
    x, edge_list = graph.x, graph.edge_index
    G = nx.Graph()
    edge_list = graph.edge_index.t().tolist()
    G.add_edges_from(edge_list)

    num_nodes = G.number_of_nodes()

    depths = []
    for node in G.nodes:
        lengths = nx.single_source_shortest_path_length(G, node)
        depths.append(max(lengths.values()))
    if len(depths) == 0:
        max_depth = 0
    else:
        max_depth = max(depths)
    #avg_depth = sum(depths) / len(depths)
    
    width_levels = []
    for node in G.nodes:
        level_count = list(nx.single_source_shortest_path_length(G, node).values())
        width_levels.append(max(level_count, key=level_count.count))
    if len(width_levels) == 0:
        max_width = 0
    else:
        max_width = max(width_levels)
    #avg_width = sum(width_levels) / len(width_levels)

    if my_new_mask_idx in G.nodes:
        lengths = nx.single_source_shortest_path_length(G, 0)  # Assuming node 0 is the root
        depth_to_my_node = lengths.get(my_new_mask_idx, -1)  # -1 if not reachable
    else:
        depth_to_my_node = -1
    
    # Compute hop neighborhoods from my_new_mask_idx
    hop_counts = {k: 0 for k in range(1, 6)}
    if my_new_mask_idx in G:
        lengths = nx.single_source_shortest_path_length(G, my_new_mask_idx)
        for k in range(1, 6):
            hop_counts[k] = sum(1 for v in lengths.values() if v <= k)
    

    return num_nodes, max_depth, max_width, depth_to_my_node, hop_counts


def analyze_graph(data):
    if data is None: 
        return

    # Apply trimming
    mask = data["y_mask"]

    _, edges_dic_num, conv_indices_to_keep, my_new_mask_idx = get_graph(data.x_text, mask, with_temporal_edges=False, undirected=False, trim=False, new_trim=False)
    num_nodes_before, max_depth_before, max_width_before, depth_to_my_node_before, hop_counts_before = graph_stats(data, edges_dic_num, conv_indices_to_keep, my_new_mask_idx)

    _, edges_dic_num, conv_indices_to_keep, my_new_mask_idx= get_graph(data.x_text, mask, with_temporal_edges=False, undirected=False, trim=True, new_trim=False)
    num_nodes_before, max_depth_before, max_width_before, depth_to_my_node_before, hop_counts_before = graph_stats(data, edges_dic_num, conv_indices_to_keep, my_new_mask_idx)

    num_nodes_after, max_depth_after, max_width_after, depth_to_my_node_after, hop_counts_after = graph_stats(data, edges_dic_num, conv_indices_to_keep, my_new_mask_idx)
    return num_nodes_before, max_depth_before, max_width_before, depth_to_my_node_before, hop_counts_before, num_nodes_after, max_depth_after, max_width_after, depth_to_my_node_after, hop_counts_after


def analyze_graphs(loader, output_csv="output_graphs.csv"):
    all_depths_before, all_widths_before, all_depths_to_my_node_before, all_nodes_before, all_depths_after, all_widths_after, all_depths_to_my_node_after, all_nodes_after = [], [], [], [], [], [], [], []
    neighborhood_counts_before = {k: [] for k in range(1, 6)}
    neighborhood_counts_after = {k: [] for k in range(1, 6)}  # For 1 to 5-hop neighbors
    i = 0
    data_list = []
    for data in loader:
        print('data sample number: ', i)
        i = i + 1
        # Extract graph before trimming
        num_nodes_before, max_depth_before, max_width_before, depth_to_my_node_before, hop_counts_before, num_nodes_after, max_depth_after, max_width_after, depth_to_my_node_after, hop_counts_after = analyze_graph(data) 
        all_nodes_before.append(num_nodes_before)
        all_nodes_after.append(num_nodes_after)
        all_depths_before.append(max_depth_before)
        all_depths_after.append(max_depth_after)
        all_widths_before.append(max_width_before)
        all_widths_after.append(max_width_after)
        all_depths_to_my_node_before.append(depth_to_my_node_before)
        all_depths_to_my_node_after.append(depth_to_my_node_after)
        for k in range(1, 6):
            neighborhood_counts_before[k].append(hop_counts_before[k])
            neighborhood_counts_after[k].append(hop_counts_after[k])
        # Prepare row for CSV
        row = {
            "Sample": i,
            "Nodes Before": num_nodes_before,
            "Depth Before": max_depth_before,
            "Width Before": max_width_before,
            "Depth to My Node Before": depth_to_my_node_before,
            "Nodes After": num_nodes_after,
            "Depth After": max_depth_after,
            "Width After": max_width_after,
            "Depth to My Node After": depth_to_my_node_after,
        }
        
        # Add hop counts (1 to 5-hop neighborhoods before and after trimming)
        for k in range(1, 6):
            row[f"Hop {k} Before"] = hop_counts_before[k]
            row[f"Hop {k} After"] = hop_counts_after[k]
        
        # Append data to list
        data_list.append(row)
    
        
    
    # Compute statistics
    stats = {
        "Nodes Before Trimming": {"Min": min(all_nodes_before), "Max": max(all_nodes_before), "Avg": sum(all_nodes_before) / len(all_nodes_before)},
        "Graph Depth Before Trimming": {"Min": min(all_depths_before), "Max": max(all_depths_before), "Avg": sum(all_depths_before) / len(all_depths_before)},
        "Graph Width Before Trimming": {"Min": min(all_widths_before), "Max": max(all_widths_before), "Avg": sum(all_widths_before) / len(all_widths_before)},
        "Neighborhood Counts Before Trimming": {k: sum(neighborhood_counts_before[k]) / len(neighborhood_counts_before[k]) for k in range(1, 6)},


        "Nodes After Trimming": {"Min": min(all_nodes_after), "Max": max(all_nodes_after), "Avg": sum(all_nodes_after) / len(all_nodes_after)},
        "Graph Depth After Trimming": {"Min": min(all_depths_after), "Max": max(all_depths_after), "Avg": sum(all_depths_after) / len(all_depths_after)},
        "Graph Width After Trimming": {"Min": min(all_widths_after), "Max": max(all_widths_after), "Avg": sum(all_widths_after) / len(all_widths_after)},
        "Neighborhood Counts After Trimming": {k: sum(neighborhood_counts_after[k]) / len(neighborhood_counts_after[k]) for k in range(1, 6)}
    }
    # Convert to DataFrame
    df = pd.DataFrame(data_list)

    # Save to CSV
    df.to_csv(output_csv, index=False)

    print(f"Results saved to {output_csv}")
    
    return stats


if __name__ == "__main__":
    train_loader, valid_loader, test_loader = get_graph_dataloaders("cad", True, 0) 

    print("======================== TRAIN SET CAD ===============================")
    stats = analyze_graphs(train_loader, 'train_graphs_cad.csv')
    print(stats)
    print("======================== TEST SET CAD ===============================")
    stats = {}
    stats = analyze_graphs(train_loader, 'test_graphs_cad.csv')    
    print(stats)
