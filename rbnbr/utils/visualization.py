from matplotlib.legend_handler import HandlerTuple



import networkx as nx
import matplotlib.pyplot as plt




import matplotlib




import matplotlib.lines as mlines
import numpy as np

def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    
    
    
    
def draw_tsp_solution(G, order, colors, pos):
    G2 = nx.DiGraph()
    G2.add_nodes_from(G)
    n = len(order)
    for i in range(n):
        j = (i + 1) % n
        G2.add_edge(order[i], order[j], weight=G[order[i]][order[j]]["weight"])
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(
        G2, node_color=colors, edge_color="b", node_size=600, alpha=0.8, ax=default_axes, pos=pos
    )
    edge_labels = nx.get_edge_attributes(G2, "weight")
    nx.draw_networkx_edge_labels(G2, pos, font_color="b", edge_labels=edge_labels)



def visualize_cut(edge_list=None, graph=None, solution=None, pos=None, opt={}, ax=None, title=None, show_edge_labels=False, save_svg=None, use_title=True, save_pgf=None, show_node_labels=True, edge_width=0.2, non_cut_edge_width=0.0):
    c0 = 'red'
    c1 = 'blue'
    
    # Create a graph from the edge list
    if graph is None:
        G = nx.Graph()
        for u, v, weight in edge_list:
            G.add_edge(u, v, weight=weight)
    else:
        G = graph

    # Set up the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    if pos is None:
        pos = nx.spring_layout(G)

    # Draw all nodes with same color if no solution provided
    if solution is None:
        node_size = opt.get('node_size', 1800)
        node_color = opt.get('node_color', 'black')
        nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, ax=ax, alpha=0.5)
    else:
        # Determine the partition based on the solution
        partition_0 = [i for i, val in enumerate(solution) if val == -1 or val == 0]
        partition_1 = [i for i, val in enumerate(solution) if val == 1]

        # Draw the nodes with different colors
        node_size = opt.get('node_size', 1800)
        node_color_0 = opt.get('node_color', c0)
        node_color_1 = opt.get('node_color_alt', c1)
        nx.draw_networkx_nodes(G, pos, nodelist=partition_0, node_color=node_color_0, node_size=node_size, ax=ax, alpha=0.5)
        nx.draw_networkx_nodes(G, pos, nodelist=partition_1, node_color=node_color_1, node_size=node_size, ax=ax, alpha=0.5)

    # Draw the edges
    edge_alpha = opt.get('edge_alpha', 1.0)
    if solution is None:
        # Draw all edges as non-cut edges
        non_cut_edge_color = opt.get('edge_color', 'black')
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color=non_cut_edge_color, width=edge_width, ax=ax, alpha=edge_alpha)
    else:
        # Draw edges based on solution
        cut_edges = [(u, v) for (u, v, _) in G.edges(data=True) if solution[u] != solution[v]]
        non_cut_edges0 = [(u, v) for (u, v, _) in G.edges(data=True) if solution[u] == solution[v] and solution[u] == 0]
        non_cut_edges1 = [(u, v) for (u, v, _) in G.edges(data=True) if solution[u] == solution[v] and solution[u] == 1]
        
        # cut edges: solid grey line
        cut_edge_color = opt.get('cut_edge_color', 'grey')
        nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color=cut_edge_color, width=edge_width, ax=ax, alpha=edge_alpha)
        
        non_cut_edge_color0 = opt.get('edge_color', c0)
        non_cut_edge_color1 = opt.get('edge_color_alt', c1)
        
        nx.draw_networkx_edges(G, pos, edgelist=non_cut_edges0, edge_color=non_cut_edge_color0, width=non_cut_edge_width, ax=ax, alpha=edge_alpha)
        nx.draw_networkx_edges(G, pos, edgelist=non_cut_edges1, edge_color=non_cut_edge_color1, width=non_cut_edge_width, ax=ax, alpha=edge_alpha)

    # Add labels to the nodes if show_node_labels is True
    if show_node_labels:
        font_size = opt.get('font_size', 6)
        font_color = opt.get('font_color', 'white')
        nx.draw_networkx_labels(G, pos, font_size=font_size, font_color=font_color, ax=ax)

    # Add edge labels (weights) only if not all weights are 1
    show_edge_labels = show_edge_labels and not all(G.edges[edge]['weight'] == 1 for edge in G.edges())
    edge_labels = nx.get_edge_attributes(G, 'weight')
    if solution is not None:
        for edge, label in edge_labels.items():
            if edge in non_cut_edges0 or edge in non_cut_edges1:
                edge_labels[edge] = ''
                
    if not all(weight == 1 for weight in edge_labels.values()) and show_edge_labels:
        # Create small rounded box for edge labels
        bbox = dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.6)
        nx.draw_networkx_edge_labels(
            G, pos, 
            edge_labels=edge_labels, 
            ax=ax,
            font_size=5,  # Smaller font size
            bbox=bbox,    # Rounded box
            label_pos=0.5 # Center label on edge
        )

    if use_title:
        ax.set_title(f"Max-Cut Visualization: {title}")
    # Add legend for edges

    if solution is not None:
        # Create custom line segments for legend
        cut_line = mlines.Line2D([], [], color=opt.get('cut_edge_color', 'grey'), linewidth=edge_width, label='Cut edges')
        
        # Create legend elements as individual lines
        legend_elements = [cut_line]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    ax.axis('off')
    
    fig = plt.gcf()
    
    plt.tight_layout()
    plt.show()

    if save_svg is not None:
        fig.savefig(save_svg, format="svg")
    if save_pgf is not None:
        original_backend = matplotlib.get_backend()
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
        fig.savefig(save_pgf)
        matplotlib.use(original_backend)
        
    return pos


def plot_result_distribution(final_distribution_bin, best_bitstring):
    matplotlib.rcParams.update({"font.size": 10})
    final_bits = final_distribution_bin
    values = np.abs(list(final_bits.values()))
    top_4_values = sorted(values, reverse=True)[:4]
    positions = []
    for value in top_4_values:
        found_positions = np.where(values == value)[0]
        if isinstance(found_positions, np.ndarray):
            positions.extend(found_positions.tolist())
        else:
            positions.append(found_positions)
    positions = positions[:4]  # Ensure we only keep the top 4 positions
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.xticks(rotation=45)
    plt.title("Result Distribution")
    plt.xlabel("Bitstrings (reversed)")
    plt.ylabel("Probability")
    ax.bar(list(final_bits.keys()), list(final_bits.values()), color="tab:grey")
    for p in positions:
        ax.get_children()[int(p)].set_color("tab:purple")

    # Set the x-label color to red for the best_bitstring
    for label in ax.get_xticklabels():
        if label.get_text() == best_bitstring:
            label.set_color('red')

    plt.show()
    
    

def show_optim_history(y, exp_val, optim_params):
    print(f"Expected value: {exp_val:.4f}")
    print(f"Optimized parameters: {optim_params}")
    if len(y) > 0:
        plt.figure(figsize=(10, 3))
        x = list(range(1, len(y)+1))
        plt.plot(x, y, marker='o', linestyle='-', color='b')
        plt.xlabel('Iteration')
        plt.ylabel('Expectation Value')
        plt.title('Optimization History')
        plt.grid(True)
        plt.xticks(x)
        plt.show()
    else:
        print("Parameters Assigned: No optimization history")