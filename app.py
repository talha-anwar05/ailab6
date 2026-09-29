import streamlit as st
import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="Search Visualizer", layout="centered")

# --- DATA ---
locations = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6)
}

hospital_graph = {
    "Pharmacy": {
        "Main_Corridor": 2.2,
        "Patient_Wing": 4.1
    },
    "Main_Corridor": {
        "Nursing_Station": 2.2
    },
    "Patient_Wing": {
        "Laboratory": 5.0,
        "Nursing_Station": 3.2
    },
    "Nursing_Station": {
        "Laboratory": 3.2,
        "Emergency_Ward": 6.0
    },
    "Laboratory": {
        "Emergency_Ward": 3.2
    },
    "Emergency_Ward": {}
}

# --- ALGORITHMS ---
def heuristic(current, goal):
    x1, y1 = locations[current]
    x2, y2 = locations[goal]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def greedy_best_first_search(start, goal, graph):
    pq = []
    heapq.heappush(pq, (heuristic(start, goal), start))
    visited = set()
    came_from = {}
    expansion_order = []
    
    while pq:
        h, current = heapq.heappop(pq)
        
        if current in visited:
            continue
            
        visited.add(current)
        expansion_order.append(current)
        
        if current == goal:
            break
            
        for neighbor in graph[current]:
            if neighbor not in visited:
                heapq.heappush(pq, (heuristic(neighbor, goal), neighbor))
                if neighbor not in came_from:
                    came_from[neighbor] = current

    path = []
    curr = goal
    while curr != start:
        path.append(curr)
        curr = came_from.get(curr)
        if curr is None:
            break
    path.append(start)
    path.reverse()

    total_cost = 0
    for i in range(len(path) - 1):
        total_cost += graph[path[i]][path[i+1]]

    return path, total_cost

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    path.reverse()
    return path

def a_star_search(start, goal, graph):
    pq = []
    heapq.heappush(pq, (heuristic(start, goal), start))
    g_cost = {start: 0}
    came_from = {}
    visited = set()
    
    while pq:
        f, current = heapq.heappop(pq)
        
        if current in visited:
            continue
            
        visited.add(current)
        
        if current == goal:
            break
            
        for neighbor, cost in graph[current].items():
            tentative_g_cost = g_cost[current] + cost
            if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g_cost
                f_cost = tentative_g_cost + heuristic(neighbor, goal)
                heapq.heappush(pq, (f_cost, neighbor))
                came_from[neighbor] = current

    if goal not in g_cost:
        return [start], 0

    path = reconstruct_path(came_from, goal)
    total_cost = g_cost[goal]

    return path, total_cost


# --- UI ---
st.title("🤖 Interactive Search Visualization")
st.markdown("Select your starting point, destination, and the algorithm you want to use. Then click **Run Search** to visualize the path!")

st.sidebar.header("Configuration")
node_list = list(locations.keys())
start_node = st.sidebar.selectbox("Start Node", node_list, index=0)
goal_node = st.sidebar.selectbox("Goal Node", node_list, index=len(node_list)-1)
algorithm = st.sidebar.selectbox("Search Algorithm", ["GBFS", "A*"])

if st.sidebar.button("Run Search", type="primary"):
    if start_node == goal_node:
        st.warning("Start node and Goal node cannot be the same!")
    else:
        # Run selected algorithm
        if algorithm == "GBFS":
            path, cost = greedy_best_first_search(start_node, goal_node, hospital_graph)
        else:
            path, cost = a_star_search(start_node, goal_node, hospital_graph)
        
        # Display Results
        st.success(f"**Algorithm:** {algorithm}")
        st.info(f"**Path Found:** {' ➜ '.join(path)}")
        st.info(f"**Total Cost:** {cost:.2f}")

        # Visualization
        st.markdown("### Network Graph")
        
        G = nx.DiGraph()
        for node, neighbors in hospital_graph.items():
            for neighbor, edge_cost in neighbors.items():
                G.add_edge(node, neighbor, weight=edge_cost)

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Draw everything
        nx.draw_networkx_nodes(G, locations, node_color="#4F8BF9", node_size=1000, ax=ax)
        
        # Offset labels so the long names are completely visible outside the node
        pos_labels = {node: (coords[0], coords[1] + 0.35) for node, coords in locations.items()}
        nx.draw_networkx_labels(G, pos_labels, font_size=10, font_weight="bold", font_color="black", ax=ax)
        
        nx.draw_networkx_edges(G, locations, arrows=True, arrowsize=20, ax=ax, edge_color="gray")
        
        edge_labels = nx.get_edge_attributes(G, "weight")
        # Add background box to edge labels so they sit clearly over the lines
        nx.draw_networkx_edge_labels(
            G, locations, edge_labels=edge_labels, ax=ax,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9)
        )
        
        # Highlight solution
        if path and len(path) > 1:
            solution_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(
                G, locations, edgelist=solution_edges, edge_color="#FF4B4B", width=3, arrows=True, arrowsize=25, ax=ax
            )
        
        ax.axis("off")
        st.pyplot(fig)
else:
    st.info("Configure your search in the sidebar and click 'Run Search' to visualize the results.")
