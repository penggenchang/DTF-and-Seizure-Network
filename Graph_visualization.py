import networkx as nx
import numpy as np
import pandas as pd
import itertools

import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from matplotlib.cm import ScalarMappable

### obtain the DTF matrix
import DTF_calculation
matrix_total = DTF_calculation.DTF_summation()

def Sparsity(input_matrix,quantile):
    matrix_beta = input_matrix
    q = quantile ## to discard how many q% low edges
    df_filtered = matrix_beta.mask((matrix_beta < matrix_beta.quantile(q=0.75)))
    # then mask diagonal
    np.fill_diagonal(df_filtered.values,0)
    df_filtered = df_filtered.fillna(0)
    return df_filtered


df_filtered = Sparsity(input_matrix=matrix_total,quantile=0.75)  ## assume
#print(df_filtered)

Left_channels, Right_channels, notable_channels = DTF_calculation.Basic_information()
num_nodes = len(notable_channels)

def Node_strength(input_df):
    values_in = []
    values_out = []
    for k in range(num_nodes):
        flow_out = input_df.iloc[:, k].sum()
        values_out.append(flow_out)
        flow_in = input_df.iloc[k, :].sum()
        values_in.append(flow_in)

    #degree_out = dict(zip(notable_channels, values_out))
    #degree_in = dict(zip(notable_channels, values_in))
    #degree_out, degree_in
    return values_out, values_in


#degree_out, degree_in = Node_strength(input_df = df_filtered)

def Edge_assignment():
    ## directional graph
    G = nx.DiGraph()

    # get node position. This can be self-adjusted
    # searching from column

    position = [(2,2),(2,1.5),(1.5,2),(1.5,2.5),(1.5,1.5),(1.5,1),(1,2),(1,2.5),(1,1.5),(1,1),
    (3, 2), (3, 1.5), (3.5, 2), (3.5, 2.5), (3.5, 1.5), (3.5, 1), (4, 2), (4, 2.5), (4, 1.5), (4, 1)]

    degree_out, degree_in = Node_strength(input_df=df_filtered)
    strength_list = np.add(degree_in,degree_out).tolist()

    ## use different color to differentiate the edges
    color_dict = {'LL': 'black', 'LR': 'red', 'RR': 'blue', 'RL': 'green'}
    pos_dict = dict(zip(notable_channels, position))
    EDGES = []
    Size_node = []

    for c in notable_channels:
        pos_idx = notable_channels.index(c)
        node_size = 50 * strength_list[pos_idx]
        pos_c = position[pos_idx]
        Size_node.append(node_size)
        G.add_node(c,pos=pos_c)

        for n in notable_channels:
            if df_filtered.loc[c,n] != 0:
                value = df_filtered.loc[c,n]
                temp_dict = {'weight':value}
                edge_tuple = (c,n,temp_dict)
                EDGES.append(edge_tuple)
                # to determine edge color
                if (c in Left_channels) & (n in Left_channels):
                    color_select = color_dict['LL']
                elif (c in Left_channels) & (n in Right_channels):
                    color_select = color_dict['LR']
                elif (c in Right_channels) & (n in Right_channels):
                    color_select = color_dict['RR']
                else:
                    color_select = color_dict['RL']
                G.add_edge(c,n,weight=value,width=value*250,color=color_select)

    # G[][] is a dictionary, key is 'weight' and 'width'
    edges = G.edges()

    colors = nx.get_edge_attributes(G,'color').values()
    #nodes_selected = ['LT1','LU1','LU2','LU7','RB1','RB2']  # for 2100
    nodes_selected = ['LC1','LB1','LB2']

    nodesize_map = []
    color_map = []
    ### if want to highlight specific nodes, their size is 1000, color becomes red
    for node in G:
        if node in nodes_selected:
            color_map.append('red')
            nodesize_map.append(1000)
        else:
            color_map.append('orange')
            nodesize_map.append(440)

    nx.draw(G, with_labels=True, pos=pos_dict, edge_color=colors, node_size=nodesize_map, node_color=color_map)

    plt.show()
    return G


G = Edge_assignment()

