import numpy as np
import pandas as pd
import itertools

import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from matplotlib.cm import ScalarMappable
import seaborn as sns

### obtain the DTF matrix
import DTF_calculation

#matrix_total = DTF_calculation.DTF_summation()

def draw_DTF(input_matrix):
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    input_matrix = input_matrix.round(decimals=3)
    np.fill_diagonal(input_matrix.values, 0)
    #pd.options.display.float_format= lambda x: '{:.0f}'.format(x) if round(x,0) ==x else '{:,.3f}'.format(x)

    pd.set_option("display.precision",3)

    #ax = sns.heatmap(data=input_matrix.T,annot=True,cmap='OrRd',fmt='.1f',cbar_kws={'format':'%.1f'},annot_kws={'weight':'bold'})#,cbar=False)
    ## heatmap with color bar
    ax = sns.heatmap(data=input_matrix.T, annot=True, vmin=0, vmax=9, cmap='OrRd', fmt='.1f',cbar_kws={'format':'%.1f'})
    cbar = ax.collections[0].colorbar

    cbar.set_ticks([0,3,6,9])
    #ax = sns.heatmap(data=input_matrix, annot=True,vmin=0,vmax=1,cmap='OrRd')
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
    #ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(labelsize=12)
    plt.yticks(rotation=0)
    plt.show()

    return 1