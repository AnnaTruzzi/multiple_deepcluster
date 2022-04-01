import os
import scipy.io
import pickle
import numpy as np
from itertools import compress
from itertools import combinations
from scipy.spatial import distance
from sklearn.manifold import MDS
from statsmodels.stats.anova import AnovaRM
import matplotlib
from matplotlib import pyplot as plt
import re
from scipy.cluster.hierarchy import dendrogram, linkage
from itertools import combinations
import scipy.io
import h5py
import hdf5storage
from scipy import stats
from scipy.spatial.distance import squareform
import collections
import re
import skbio
import seaborn as sns
import pingouin as pg
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
import scipy
from scipy.stats import pearsonr, spearmanr, kendalltau
import seaborn as sns


def corr_with_brain_plot(dataframe):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    sns.lineplot(x="layer", y="corr", hue="net_type", data=dataframe[dataframe['ROI']=='EVC'], markers=True, dashes=False, ax=ax[0])    
    ax[0].set_title('Comparison to EVC')
    ax[0].axhline(0.38, color='grey', lw=2, alpha=0.4)
    ax[0].axhline(0.31, color='gray', lw=2, alpha=0.4)
    ax[0].axhspan(0.31, 0.38, facecolor='gray', alpha=0.4)

    sns.lineplot(x="layer", y="corr", hue="net_type", data=dataframe[dataframe['ROI']=='IT'], markers=True, dashes=False, ax=ax[1])    
    ax[1].set_title('Comparison to IT')
    ax[1].axhline(0.28, color='gray', lw=2, alpha=0.4)
    ax[1].axhline(0.42, color='gray', lw=2, alpha=0.4)
    ax[1].axhspan(0.28, 0.42, facecolor='gray', alpha=0.4)

    plt.savefig(f'/home/annatruzzi/multiple_deepcluster/figures/comparison_to_alexnet.png')


if __name__ == '__main__':
    corr_dc = pd.read_csv(f'/home/annatruzzi/multiple_deepcluster/results/corr_dc_layer_average.csv')
    corr_alexnet = pd.read_csv(f'/home/annatruzzi/multiple_deepcluster/results/corr_alexnet_layer_average.csv')
    corr_df = pd.concat([corr_dc,corr_alexnet])
    corr_df['net_type'] = corr_df['net']+corr_df['state']
    corr_with_brain_plot(corr_df[corr_df['net_type']!='alexnetrandomstate'])
    #corr_with_brain_anova(corr_df,method)
