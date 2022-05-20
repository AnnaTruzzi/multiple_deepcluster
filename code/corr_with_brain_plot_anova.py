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
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon

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
    costum_palette = [sns.xkcd_rgb['ocean blue'],sns.xkcd_rgb['orange']]
    sns.set_palette(costum_palette)
    sns.violinplot(x='layer', y='corr', hue='state', data=corr_df[corr_df['ROI']=='EVC'], ax=ax[0,0])
    ax[0,0].set_title('Comparison to EVC')
    if 'mri_variability' in method:
        ax[0,0].set_ylim((-0.15,0.45))
        ax[0,0].axhline(0.38, color='grey', lw=2, alpha=0.4)
        ax[0,0].axhline(0.31, color='gray', lw=2, alpha=0.4)
        ax[0,0].axhspan(0.31, 0.38, facecolor='gray', alpha=0.4)
    else:
        layers_noise_ceiling(ax[0,0])
        ax[0,0].set_ylim((-0.05,1.05))

    sns.violinplot(x='layer', y='corr', hue='state', data=corr_df[corr_df['ROI']=='IT'], ax=ax[0,1])
    ax[0,1].set_title('Comparison to IT')
    if 'mri_variability' in method:
        ax[0,1].set_ylim((-0.15,0.45))
        ax[0,1].axhline(0.28, color='gray', lw=2, alpha=0.4)
        ax[0,1].axhline(0.42, color='gray', lw=2, alpha=0.4)
        ax[0,1].axhspan(0.28, 0.42, facecolor='gray', alpha=0.4)
    else:
        layers_noise_ceiling(ax[0,1])
        ax[0,1].set_ylim((-0.05,1.05))

    sns.lineplot(x="layer", y="corr", hue="state", data=corr_df[corr_df['ROI']=='EVC'], markers=True, dashes=False, ax=ax[1,0], legend=False)    
    ax[1,0].set_title('Comparison to EVC')
    if 'mri_variability' in method:
        ax[1,0].set_ylim((-0.015,0.45))
        ax[1,0].axhline(0.38, color='grey', lw=2, alpha=0.4)
        ax[1,0].axhline(0.31, color='gray', lw=2, alpha=0.4)
        ax[1,0].axhspan(0.31, 0.38, facecolor='gray', alpha=0.4)
    else:
        layers_noise_ceiling(ax[1,0])
        ax[1,0].set_ylim((-0.05,1.05))

    sns.lineplot(x="layer", y="corr", hue="state", data=corr_df[corr_df['ROI']=='IT'], markers=True, dashes=False, ax=ax[1,1], legend=False)    
    ax[1,1].set_title('Comparison to IT')
    if 'mri_variability' in method:
        ax[1,1].set_ylim((-0.015,0.45))
        ax[1,1].axhline(0.28, color='gray', lw=2, alpha=0.4)
        ax[1,1].axhline(0.42, color='gray', lw=2, alpha=0.4)
        ax[1,1].axhspan(0.28, 0.42, facecolor='gray', alpha=0.4)
    else:
        layers_noise_ceiling(ax[1,1])
        ax[1,1].set_ylim((-0.05,1.05))

    if "layer_variability" in method:
        ax[1,0].text(3, 0.60, "**", ha='center', va='bottom', fontsize=15)
        ax[1,1].text(3, 0.60, "**", ha='center', va='bottom', fontsize=15)
    else:
        ax[1,0].text(0, 0.15, "***", ha='center', va='bottom', fontsize=10)
        ax[1,0].text(1, 0.20, "***", ha='center', va='bottom', fontsize=10)
        ax[1,0].text(2, 0.18, "**", ha='center', va='bottom', fontsize=10)
        ax[1,0].text(3, 0.18, "***", ha='center', va='bottom', fontsize=10)
        ax[1,0].text(4, 0.18, "***", ha='center', va='bottom', fontsize=10)
        ax[1,0].text(5, 0.18, "**", ha='center', va='bottom', fontsize=10)
        ax[1,0].text(6, 0.18, "*", ha='center', va='bottom', fontsize=10)

        ax[1,1].text(0, 0.15, "***", ha='center', va='bottom', fontsize=10)        
        ax[1,1].text(1, 0.18, "***", ha='center', va='bottom', fontsize=10)        
        ax[1,1].text(2, 0.18, "***", ha='center', va='bottom', fontsize=10)        
        ax[1,1].text(3, 0.18, "**", ha='center', va='bottom', fontsize=10)        
 
      
    plt.suptitle(f'{method}')
    plt.savefig(f'/home/annatruzzi/multiple_deepcluster/figures/corr_{method}.png')
    plt.savefig(f'/home/annatruzzi/multiple_deepcluster/figures/corr_{method}.pdf')


def corr_with_brain_check_dist(corr_df,method):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
    sns.distplot(corr_df[corr_df['ROI']=='EVC']['corr'],ax=ax[0])
    sns.distplot(corr_df[corr_df['ROI']=='IT']['corr'],ax=ax[1])
    ax[0].set_title('EVC')
    ax[1].set_title('IT')
    plt.suptitle(f'{method}')
    plt.savefig(f'/home/annatruzzi/multiple_deepcluster/figures/corr_dist_{method}.png')


def corr_with_brain_anova(corr_df,method,flag):
    random_1 = corr_df[(corr_df['ROI']==flag) & (corr_df['state']=='randomstate') & (corr_df['layer']=='ReLu1')]['corr']
    random_2 = corr_df[(corr_df['ROI']==flag) & (corr_df['state']=='randomstate') & (corr_df['layer']=='ReLu2')]['corr']
    random_3 = corr_df[(corr_df['ROI']==flag) & (corr_df['state']=='randomstate') & (corr_df['layer']=='ReLu3')]['corr']
    random_4 = corr_df[(corr_df['ROI']==flag) & (corr_df['state']=='randomstate') & (corr_df['layer']=='ReLu4')]['corr']
    random_5 = corr_df[(corr_df['ROI']==flag) & (corr_df['state']=='randomstate') & (corr_df['layer']=='ReLu5')]['corr']
    random_6 = corr_df[(corr_df['ROI']==flag) & (corr_df['state']=='randomstate') & (corr_df['layer']=='ReLu6')]['corr']
    random_7 = corr_df[(corr_df['ROI']==flag) & (corr_df['state']=='randomstate') & (corr_df['layer']=='ReLu7')]['corr']

    trained_1 = corr_df[(corr_df['ROI']==flag) & (corr_df['state']=='100epochs') & (corr_df['layer']=='ReLu1')]['corr']
    trained_2 = corr_df[(corr_df['ROI']==flag) & (corr_df['state']=='100epochs') & (corr_df['layer']=='ReLu2')]['corr']
    trained_3 = corr_df[(corr_df['ROI']==flag) & (corr_df['state']=='100epochs') & (corr_df['layer']=='ReLu3')]['corr']
    trained_4 = corr_df[(corr_df['ROI']==flag) & (corr_df['state']=='100epochs') & (corr_df['layer']=='ReLu4')]['corr']
    trained_5 = corr_df[(corr_df['ROI']==flag) & (corr_df['state']=='100epochs') & (corr_df['layer']=='ReLu5')]['corr']
    trained_6 = corr_df[(corr_df['ROI']==flag) & (corr_df['state']=='100epochs') & (corr_df['layer']=='ReLu6')]['corr']
    trained_7 = corr_df[(corr_df['ROI']==flag) & (corr_df['state']=='100epochs') & (corr_df['layer']=='ReLu7')]['corr']
    print(flag)
    stats = kruskal(random_1,random_2,random_3,random_4,random_5,random_6,random_7,trained_1,trained_2,trained_3,trained_4,trained_5,trained_6,trained_7)
    print(stats)
    #model=ols('corr ~ C(layer) + C(state) + C(layer):C(state)', data=corr_df[corr_df['ROI']=='EVC']).fit() #Specify C for Categorical
    #print(sm.stats.anova_lm(model, typ=2))
    print('Testing for normality...')
    print(scipy.stats.shapiro(corr_df[corr_df['ROI']=='EVC']['corr']))
    print(scipy.stats.shapiro(corr_df[corr_df['ROI']=='IT']['corr']))
    print('Testing for homoscedasticity...')
    print(pg.homoscedasticity(data=corr_df[corr_df['ROI']=='EVC']))
    print(pg.homoscedasticity(data=corr_df[corr_df['ROI']=='IT']))
    return stats


def posthoc_tests(corr_df,method,layer,flag):
        print(f'{flag} - {layer}')
        dist_random = corr_df[(corr_df['ROI']==flag) & (corr_df['state']=='randomstate') & (corr_df['layer']==layer)]['corr']
        dist_trained = corr_df[(corr_df['ROI']==flag) & (corr_df['state']=='100epochs') & (corr_df['layer']==layer)]['corr']
        res = wilcoxon(dist_random, dist_trained)
        print(res)
        return res


if __name__ == '__main__':
    corr_methods = ['layer_variability','mri_variability']
    layers = ['ReLu1','ReLu2','ReLu3','ReLu4','ReLu5','ReLu6','ReLu7']
    ROIs = ['EVC','IT']
    for method in corr_methods:
        corr_df = pd.read_csv(f'/home/annatruzzi/multiple_deepcluster/results/corr_dc_{method}.csv')
        corr_with_brain_plot(corr_df,method)
        corr_with_brain_check_dist(corr_df,method)
        with open (f'/home/annatruzzi/multiple_deepcluster/results/corr_to_brain_analysis_{method}.txt','w') as f:
            for ROI in ROIs:
                f.write(f'\n ####### {ROI} \n')
                f.write('KRUSKAL test \n')
                stat_kruskal=corr_with_brain_anova(corr_df,method,flag=ROI)
                f.write(f'{stat_kruskal} \n')
                f.write('Wilcoxon test \n')
                for layer in layers:
                    posthoc = posthoc_tests(corr_df,method,layer,flag=ROI)
                    f.write(f'{layer}: {posthoc} \n')