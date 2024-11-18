import networkx as nx
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from matplotlib.cm import ScalarMappable

### obtain the DTF matrix
import DTF_calculation
matrix_total = DTF_calculation.DTF_summation()

sz_series, frame_count = DTF_calculation.Data_split()
Left_channels, Right_channels, notable_channels = DTF_calculation.Basic_information()
## calculate the graph density

def Network_features_2(input_df):
    dtf_bandmatrix = input_df
    # mask diagonal
    np.fill_diagonal(dtf_bandmatrix.values, 0)
    dtf_filtered = dtf_bandmatrix.fillna(0)
    dtf_filtered = dtf_bandmatrix.to_numpy()
    # calculate network density
    # sub_network
    dim_left = int(len(Left_channels))
    dim_right = int(len(notable_channels))-dim_left
    subnetwork_left = dtf_filtered[0:dim_left,0:dim_left]
    #print(subnetwork_left)
    den_left = subnetwork_left.sum() / (dim_left* (dim_left-1))
    subnetwork_right = dtf_filtered[dim_left:,dim_left:]
    den_right = subnetwork_right.sum() / (dim_right * (dim_right - 1))
    sub_lefttoright = dtf_filtered[0:dim_left,dim_left:]
    den_lefttoright = sub_lefttoright.sum()/(dim_right*dim_left)
    sub_righttoleft = dtf_filtered[dim_left+1:,0:dim_left]
    den_righttoleft = sub_righttoleft.sum()/(dim_right*dim_left)

    left_impact = den_left
    right_impact = den_right

    return left_impact, right_impact

def Get_sequences_2(input_sequence):
    left_list = []
    right_list = []

    for frame in range(frame_count):
        input_df = input_sequence[frame]
        left_den_frame, right_den_frame = Network_features_2(input_df=input_df)
        left_list.append(left_den_frame)
        right_list.append(right_den_frame)


    return left_list, right_list

left_list, right_list = Get_sequences_2(input_sequence=sz_series)

## draw the boxplots
def Box_density():
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rc('font', **{'family': 'serif', 'serif': ['Times']})
    plt.rc('text', usetex=True)
    labelfont = FontProperties()
    labelfont.set_family('serif')
    labelfont.set_name('Times New Roman')

    labelfont.set_weight('normal')

    data_test_1 = pd.DataFrame({'L-impact': left_list, 'R-impact': right_list},
                               columns=['L-impact', 'R-impact'])
    v1 = {}  # assign data values
    v1['L-impact'] = data_test_1.loc[:, 'L-impact']
    v1['R-impact'] = data_test_1.loc[:, 'R-impact']


    controls_1 = ['L-impact','R-impact']
    colors_1 = ["black", "red", 'blue', 'green']
    # a new side-by-side boxplot
    data_to_plot = [v1['L-impact'], v1['R-impact']]
    bplt_1 = ax.boxplot(data_to_plot, positions=[0.5, 1.0], labels=['L-impact: \n(L-L + L-R)', 'R-impact: \n(R-R + R-L)'],
                        patch_artist=False, widths=0.2, showfliers=False,
                        whiskerprops={'color': "black", 'linestyle': '--'})


    for median in bplt_1['medians']:
        median.set(color='black', linewidth=4.5)

    ax.set_ylabel('Impact Values', fontsize=14, fontdict={'family': 'serif'})
    #ax.set_xlabel('Connectivity', fontsize=14, fontdict={'family': 'serif'})
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=12)
    #ax.set_yticks(np.arange(0, 2.5, 0.5))

    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)

    plt.show()
    return 1


Box_density()



