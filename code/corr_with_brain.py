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

def loadmat(matfile):
    try:
        f = h5py.File(matfile)
    except (IOError, OSError):
        return io.loadmat(matfile)
    else:
        return {name: np.transpose(f.get(name)) for name in f.keys()}


def corr_layer_average(layer, mri_data):
    x = np.mean(np.asarray(layer), axis=0)
    y = mri_data
    corr_list = []
    p_list = []
    for subj in range(0,y.shape[0]):
        y_subj = squareform(squareform(y[subj],checks = False))
        corr_value, p_value, n_value = skbio.stats.distance.mantel(x,y_subj, method = 'spearman', permutations = 10000)
        corr_list.append(corr_value)
        p_list.append(p_value)
    return corr_list,p_list


def corr_mri_average(layer, mri_data):
    x = np.array(layer)
    y = squareform(squareform(np.mean(np.asarray(mri_data),axis=0),checks = False))
    corr_list = []
    p_list = []
    for layer in range(0,x.shape[0]):
        x_layer = squareform(squareform(x[layer],checks = False))
        corr_value, p_value, n_value = skbio.stats.distance.mantel(x_layer,y, method = 'spearman', permutations = 10000)
        corr_list.append(corr_value)
        p_list.append(p_value)
    return corr_list,p_list


def main(list_file):
    fmri_mat = loadmat(fmri_pth)
    EVC = fmri_mat['EVC_RDMs']
    IT = fmri_mat['IT_RDMs']

    corr_methods = ['mri_average','layer_average']

    for method in corr_methods:
        print(f'Calculating correlations for {method}')
        all_corr=[]
        all_pvalue=[]
        out_layers_list=[]
        ROI_list=[]
        state_list=[]
        instane_list = []
        with open(f'/home/annatruzzi/multiple_deepcluster/results/corr_{method}.txt', 'w') as f:
            for layer in layers:
                for state in states:
                    all_layer_rdms = []
                    curr_list = [i for i in list_file if state in i and layer in i]
                    for name in curr_list:
                        with open((f'/data/multiple_deepcluster/rdms/{name}.pickle'), 'rb') as handle:
                            rdm = pickle.load(handle)
                        all_layer_rdms.append(rdm)
                    
                    if 'layer' in method:
                        EVC_corr_list,EVC_pvalue_list = corr_layer_average(all_layer_rdms, EVC)
                        IT_corr_list,IT_pvalue_list = corr_layer_average(all_layer_rdms, IT)
                    else:
                        EVC_corr_list,EVC_pvalue_list = corr_mri_average(all_layer_rdms, EVC)
                        IT_corr_list,IT_pvalue_list = corr_mri_average(all_layer_rdms, IT)
                    

                    all_corr.extend(EVC_corr_list+IT_corr_list)
                    all_pvalue.extend(EVC_pvalue_list+IT_pvalue_list)
                    out_layers_list.extend(np.repeat(layer,len(EVC_corr_list+IT_corr_list)))
                    ROI_list.extend(np.repeat('EVC',len(EVC_corr_list)))
                    ROI_list.extend(np.repeat('IT',len(IT_corr_list)))
                    state_list.extend(np.repeat(state,len(EVC_corr_list+IT_corr_list)))

                    f.write('******************************** \n')
                    f.write('******************************** \n')
                    f.write(f'Average corr with brain - Layer Average {layer} - {state} \n')
                    f.write(f'EVC: {np.mean(np.array(EVC_corr_list))} \n')
                    f.write(f'IT: {np.mean(np.array(IT_corr_list))} \n')

        print('Working on the plot...')
        corr_dict = {'corr': all_corr,
                    'p': all_pvalue,
                    'layer': out_layers_list,
                    'ROI': ROI_list,
                    'state': state_list}
        corr_df = pd.DataFrame(corr_dict)
        corr_df.to_csv(f'/home/annatruzzi/multiple_deepcluster/results/corr_{method}.csv')

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
        sns.violinplot(x='layer', y='corr', hue='state', data=corr_df[corr_df['ROI']=='EVC'], ax=ax[0,0])
        ax[0,0].set_title('Comparison to EVC')
        if 'layer' in method:
            ax[0,0].axhline(0.38, color='grey', lw=2, alpha=0.4)
            ax[0,0].axhline(0.31, color='gray', lw=2, alpha=0.4)
            ax[0,0].axhspan(0.31, 0.38, facecolor='gray', alpha=0.4)
        

        sns.violinplot(x='layer', y='corr', hue='state', data=corr_df[corr_df['ROI']=='IT'], ax=ax[0,1])
        ax[0,1].set_title('Comparison to IT')
        if 'layer' in method:
            ax[0,1].axhline(0.28, color='gray', lw=2, alpha=0.4)
            ax[0,1].axhline(0.42, color='gray', lw=2, alpha=0.4)
            ax[0,1].axhspan(0.28, 0.42, facecolor='gray', alpha=0.4)
        

        sns.lineplot(x="layer", y="corr", hue="state", data=corr_df[corr_df['ROI']=='EVC'], markers=True, dashes=False, ax=ax[1,0])    
        ax[1,0].set_title('Comparison to EVC')
        if 'layer' in method:
            ax[1,0].axhline(0.38, color='grey', lw=2, alpha=0.4)
            ax[1,0].axhline(0.31, color='gray', lw=2, alpha=0.4)
            ax[1,0].axhspan(0.31, 0.38, facecolor='gray', alpha=0.4)

        sns.lineplot(x="layer", y="corr", hue="state", data=corr_df[corr_df['ROI']=='IT'], markers=True, dashes=False, ax=ax[1,1])    
        ax[1,1].set_title('Comparison to IT')
        if 'layer' in method:  
            ax[1,1].axhline(0.28, color='gray', lw=2, alpha=0.4)
            ax[1,1].axhline(0.42, color='gray', lw=2, alpha=0.4)
            ax[1,1].axhspan(0.28, 0.42, facecolor='gray', alpha=0.4)

        plt.savefig(f'/home/annatruzzi/multiple_deepcluster/figures/corr_{method}.png')


if __name__ == '__main__':
    layers = ['ReLu1', 'ReLu2', 'ReLu3', 'ReLu4', 'ReLu5','ReLu6','ReLu7']
    states = ['randomstate','100epochs']
    instances = 11
    fmri_pth = '/data/algonautsChallenge2019/Training_Data/92_Image_Set/target_fmri.mat'
    list_file = []
    for instance in range(1,instances):
        for layer in layers:
            for state in states:
                list_file.append(f'dc{instance}_{state}_{layer}')
    main(list_file)

