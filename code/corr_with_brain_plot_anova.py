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
from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import ols
import statsmodels.api as sm


def layers_noise_ceiling(axes):
        #### ReLu1
        # randomstate
        axes.axhline(0.976, xmin=0, xmax=0.14, color='blue', lw=2, alpha=0.4)
        axes.axhline(0.971, xmin=0, xmax=0.14, color='blue', lw=2, alpha=0.4)
        axes.axhspan(0.971, 0.976, xmin=0, xmax=0.14, facecolor='blue', alpha=0.4)

        # 100 epochs
        axes.axhline(0.9981, xmin=0, xmax=0.14, color='orange', lw=2, alpha=0.4)
        axes.axhline(0.9977, xmin=0, xmax=0.14, color='orange', lw=2, alpha=0.4)
        axes.axhspan(0.9977, 0.9981, xmin=0, xmax=0.14, facecolor='orange', alpha=0.4)

        #### ReLu2
        # randomstate
        axes.axhline(0.977, xmin=0.14, xmax=0.28, color='blue', lw=2, alpha=0.4)
        axes.axhline(0.972, xmin=0.14, xmax=0.28, color='blue', lw=2, alpha=0.4)
        axes.axhspan(0.972, 0.977, xmin=0.14, xmax=0.28, facecolor='blue', alpha=0.4)

        # 100 epochs
        axes.axhline(0.995, xmin=0.14, xmax=0.28, color='orange', lw=2, alpha=0.4)
        axes.axhline(0.993, xmin=0.14, xmax=0.28, color='orange', lw=2, alpha=0.4)
        axes.axhspan(0.993, 0.995, xmin=0.14, xmax=0.28, facecolor='orange', alpha=0.4)

        #### ReLu3
        # randomstate
        axes.axhline(0.977, xmin=0.28, xmax=0.42, color='blue', lw=2, alpha=0.4)
        axes.axhline(0.971, xmin=0.28, xmax=0.42, color='blue', lw=2, alpha=0.4)
        axes.axhspan(0.971, 0.977, xmin=0.28, xmax=0.42, facecolor='blue', alpha=0.4)

        # 100 epochs
        axes.axhline(0.989, xmin=0.28, xmax=0.42, color='orange', lw=2, alpha=0.4)
        axes.axhline(0.986, xmin=0.28, xmax=0.42, color='orange', lw=2, alpha=0.4)
        axes.axhspan(0.986, 0.989, xmin=0.28, xmax=0.42, facecolor='orange', alpha=0.4)

        #### ReLu4
        # randomstate
        axes.axhline(0.971, xmin=0.42, xmax=0.56, color='blue', lw=2, alpha=0.4)
        axes.axhline(0.964, xmin=0.42, xmax=0.56, color='blue', lw=2, alpha=0.4)
        axes.axhspan(0.964, 0.971, xmin=0.42, xmax=0.56, facecolor='blue', alpha=0.4)

        # 100 epochs
        axes.axhline(0.954, xmin=0.42, xmax=0.56, color='orange', lw=2, alpha=0.4)
        axes.axhline(0.943, xmin=0.42, xmax=0.56, color='orange', lw=2, alpha=0.4)
        axes.axhspan(0.943, 0.954, xmin=0.42, xmax=0.56, facecolor='orange', alpha=0.4)

        #### ReLu5
        # randomstate
        axes.axhline(0.958, xmin=0.56, xmax=0.70, color='blue', lw=2, alpha=0.4)
        axes.axhline(0.949, xmin=0.56, xmax=0.70, color='blue', lw=2, alpha=0.4)
        axes.axhspan(0.949, 0.958, xmin=0.56, xmax=0.70, facecolor='blue', alpha=0.4)

        # 100 epochs
        axes.axhline(0.933, xmin=0.56, xmax=0.70, color='orange', lw=2, alpha=0.4)
        axes.axhline(0.917, xmin=0.56, xmax=0.70, color='orange', lw=2, alpha=0.4)
        axes.axhspan(0.917, 0.933, xmin=0.56, xmax=0.70, facecolor='orange', alpha=0.4)

        #### ReLu6
        # randomstate
        axes.axhline(0.989, xmin=0.70, xmax=0.84, color='blue', lw=2, alpha=0.4)
        axes.axhline(0.987, xmin=0.70, xmax=0.84, color='blue', lw=2, alpha=0.4)
        axes.axhspan(0.987, 0.989, xmin=0.70, xmax=0.84, facecolor='blue', alpha=0.4)

        # 100 epochs
        axes.axhline(0.961, xmin=0.70, xmax=0.84, color='orange', lw=2, alpha=0.4)
        axes.axhline(0.952, xmin=0.70, xmax=0.84, color='orange', lw=2, alpha=0.4)
        axes.axhspan(0.952, 0.961, xmin=0.70, xmax=0.84, facecolor='orange', alpha=0.4)

        #### ReLu7
        # randomstate
        axes.axhline(0.985, xmin=0.84, xmax=1, color='blue', lw=2, alpha=0.4)
        axes.axhline(0.982, xmin=0.84, xmax=1, color='blue', lw=2, alpha=0.4)
        axes.axhspan(0.982, 0.985, xmin=0.84, xmax=1, facecolor='blue', alpha=0.4)

        # 100 epochs
        axes.axhline(0.957, xmin=0.84, xmax=1, color='orange', lw=2, alpha=0.4)
        axes.axhline(0.947, xmin=0.84, xmax=1, color='orange', lw=2, alpha=0.4)
        axes.axhspan(0.947, 0.957, xmin=0.84, xmax=1, facecolor='orange', alpha=0.4)


def corr_with_brain_plot(corr_df,method):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    sns.violinplot(x='layer', y='corr', hue='state', data=corr_df[corr_df['ROI']=='EVC'], ax=ax[0,0])
    ax[0,0].set_title('Comparison to EVC')
    if 'layer' in method:
        ax[0,0].axhline(0.38, color='grey', lw=2, alpha=0.4)
        ax[0,0].axhline(0.31, color='gray', lw=2, alpha=0.4)
        ax[0,0].axhspan(0.31, 0.38, facecolor='gray', alpha=0.4)
    else:
        layers_noise_ceiling(ax[0,0])

    sns.violinplot(x='layer', y='corr', hue='state', data=corr_df[corr_df['ROI']=='IT'], ax=ax[0,1])
    ax[0,1].set_title('Comparison to IT')
    if 'layer' in method:
        ax[0,1].axhline(0.28, color='gray', lw=2, alpha=0.4)
        ax[0,1].axhline(0.42, color='gray', lw=2, alpha=0.4)
        ax[0,1].axhspan(0.28, 0.42, facecolor='gray', alpha=0.4)
    else:
        layers_noise_ceiling(ax[0,1])

    sns.lineplot(x="layer", y="corr", hue="state", data=corr_df[corr_df['ROI']=='EVC'], markers=True, dashes=False, ax=ax[1,0])    
    ax[1,0].set_title('Comparison to EVC')
    if 'layer' in method:
        ax[1,0].axhline(0.38, color='grey', lw=2, alpha=0.4)
        ax[1,0].axhline(0.31, color='gray', lw=2, alpha=0.4)
        ax[1,0].axhspan(0.31, 0.38, facecolor='gray', alpha=0.4)
    else:
        layers_noise_ceiling(ax[1,0])

    sns.lineplot(x="layer", y="corr", hue="state", data=corr_df[corr_df['ROI']=='IT'], markers=True, dashes=False, ax=ax[1,1])    
    ax[1,1].set_title('Comparison to IT')
    if 'layer' in method:
        ax[1,1].axhline(0.28, color='gray', lw=2, alpha=0.4)
        ax[1,1].axhline(0.42, color='gray', lw=2, alpha=0.4)
        ax[1,1].axhspan(0.28, 0.42, facecolor='gray', alpha=0.4)
    else:
        layers_noise_ceiling(ax[1,1])

    plt.suptitle(f'{method}')
    plt.savefig(f'/home/annatruzzi/multiple_deepcluster/figures/corr_{method}.png')


def corr_with_brain_check_dist(corr_df,method):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
    sns.distplot(corr_df[corr_df['ROI']=='EVC']['corr'],ax=ax[0])
    sns.distplot(corr_df[corr_df['ROI']=='IT']['corr'],ax=ax[1])
    ax[0].set_title('EVC')
    ax[1].set_title('IT')
    plt.suptitle(f'{method}')
    plt.savefig(f'/home/annatruzzi/multiple_deepcluster/figures/corr_dist_{method}.png')


def corr_with_brain_anova(corr_df,method):
    #TODO: get one corr value for each instance? Or stick with the average across layers?
    #TODO: not normal distribution, will have to change test to parametric version
    print(pg.anova(data=corr_df[corr_df['ROI']=='EVC'], dv='corr', between=['layer','state']))
    print(pg.anova(data=corr_df[corr_df['ROI']=='IT'], dv='corr', between=['layer','state']))
    print('Testing for normality...')
    print(scipy.stats.shapiro(corr_df[corr_df['ROI']=='EVC']['corr']))
    print(scipy.stats.shapiro(corr_df[corr_df['ROI']=='IT']['corr']))
    print('Testing for homoscedasticity...')
    print(pg.homoscedasticity(data=corr_df[corr_df['ROI']=='EVC']))
    print(pg.homoscedasticity(data=corr_df[corr_df['ROI']=='IT']))

if __name__ == '__main__':
    corr_methods = ['mri_average','layer_average']
    for method in corr_methods:
        corr_df = pd.read_csv(f'/home/annatruzzi/multiple_deepcluster/results/corr_{method}.csv')
        corr_with_brain_plot(corr_df,method)
        corr_with_brain_check_dist(corr_df,method)
        corr_with_brain_anova(corr_df,method)
