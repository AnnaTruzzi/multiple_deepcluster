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
from scipy.stats import pearsonr, spearmanr, kendalltau,kruskal
import seaborn as sns
from scipy.stats import mannwhitneyu

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


def corr_with_brain_anova(corr_df,flag):
    random_1 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dcrandomstate') & (corr_df['layer']=='ReLu1')]['corr']
    random_2 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dcrandomstate') & (corr_df['layer']=='ReLu2')]['corr']
    random_3 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dcrandomstate') & (corr_df['layer']=='ReLu3')]['corr']
    random_4 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dcrandomstate') & (corr_df['layer']=='ReLu4')]['corr']
    random_5 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dcrandomstate') & (corr_df['layer']=='ReLu5')]['corr']
    random_6 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dcrandomstate') & (corr_df['layer']=='ReLu6')]['corr']
    random_7 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dcrandomstate') & (corr_df['layer']=='ReLu7')]['corr']

    trained_1 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dc100epochs') & (corr_df['layer']=='ReLu1')]['corr']
    trained_2 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dc100epochs') & (corr_df['layer']=='ReLu2')]['corr']
    trained_3 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dc100epochs') & (corr_df['layer']=='ReLu3')]['corr']
    trained_4 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dc100epochs') & (corr_df['layer']=='ReLu4')]['corr']
    trained_5 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dc100epochs') & (corr_df['layer']=='ReLu5')]['corr']
    trained_6 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dc100epochs') & (corr_df['layer']=='ReLu6')]['corr']
    trained_7 = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dc100epochs') & (corr_df['layer']=='ReLu7')]['corr']

    trained_1_alexnet = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='alexnetpretrained') & (corr_df['layer']=='ReLu1')]['corr']
    trained_2_alexnet = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='alexnetpretrained') & (corr_df['layer']=='ReLu2')]['corr']
    trained_3_alexnet = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='alexnetpretrained') & (corr_df['layer']=='ReLu3')]['corr']
    trained_4_alexnet = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='alexnetpretrained') & (corr_df['layer']=='ReLu4')]['corr']
    trained_5_alexnet = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='alexnetpretrained') & (corr_df['layer']=='ReLu5')]['corr']
    trained_6_alexnet = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='alexnetpretrained') & (corr_df['layer']=='ReLu6')]['corr']
    trained_7_alexnet = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='alexnetpretrained') & (corr_df['layer']=='ReLu7')]['corr']

    print(flag)
    stats = kruskal(random_1,random_2,random_3,random_4,random_5,random_6,random_7,trained_1,trained_2,trained_3,trained_4,trained_5,trained_6,trained_7,trained_1_alexnet,
                    trained_2_alexnet,trained_3_alexnet,trained_4_alexnet,trained_5_alexnet,trained_6_alexnet,trained_7_alexnet)
    print(stats)
    #model=ols('corr ~ C(layer) + C(state) + C(layer):C(state)', data=corr_df[corr_df['ROI']=='EVC']).fit() #Specify C for Categorical
    #print(sm.stats.anova_lm(model, typ=2))
    print('Testing for normality...')
    print(scipy.stats.shapiro(corr_df[corr_df['ROI']=='EVC']['corr']))
    print(scipy.stats.shapiro(corr_df[corr_df['ROI']=='IT']['corr']))
    print('Testing for homoscedasticity...')
    print(pg.homoscedasticity(data=corr_df[corr_df['ROI']=='EVC']))
    print(pg.homoscedasticity(data=corr_df[corr_df['ROI']=='IT']))


def posthoc_tests(corr_df,flag):
    for layer in layers:
        print(f'{flag} - {layer}')
        dist_random = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='dcrandomstate') & (corr_df['layer']==layer)]['corr']
        dist_trained = corr_df[(corr_df['ROI']==flag) & (corr_df['net_type']=='alexnetpretrained') & (corr_df['layer']==layer)]['corr']
        res = mannwhitneyu(dist_random, dist_trained)
        print(res)


if __name__ == '__main__':
    layers = ['ReLu1','ReLu2','ReLu3','ReLu4','ReLu5','ReLu6','ReLu7']
    corr_dc = pd.read_csv(f'/home/annatruzzi/multiple_deepcluster/results/corr_dc_layer_average.csv')
    corr_alexnet = pd.read_csv(f'/home/annatruzzi/multiple_deepcluster/results/corr_alexnet_layer_average.csv')
    corr_df = pd.concat([corr_dc,corr_alexnet])
    corr_df['net_type'] = corr_df['net']+corr_df['state']
    ROIs = ['EVC','IT']
    corr_with_brain_plot(corr_df[corr_df['net_type']!='alexnetrandomstate'])
    for ROI in ROIs:
        corr_with_brain_anova(corr_df[corr_df['net_type']!='alexnetrandomstate'],flag=ROI)
        posthoc_tests(corr_df[(corr_df['net_type']!='alexnetrandomstate') & (corr_df['net_type']!='dc100epochs')],flag=ROI)
