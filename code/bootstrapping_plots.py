from scipy import fftpack
import numpy as np
import pylab as py
import scipy.io
from PIL import Image
import os
import pandas as pd
import glob
import re
import pickle
from scipy.spatial import distance
import matplotlib
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
from itertools import combinations
import pandas as pd
import seaborn as sns
import collections
from scipy.stats import sem


## load activations dictionary 
def load_dict(path):
    with open(path,'rb') as file:
        dict=pickle.load(file, encoding="latin1")
    return dict


def main():
    corr_values = load_dict('/home/annatruzzi/multiple_deepcluster/results/CombinedRegressionModels_BootstrappingValues_NNLS.pickle')

    grand_average = np.mean(np.array(list(corr_values.values())))
    across_model_average = np.mean(np.array(list(corr_values.values())), axis=0)
    corr_values_mean_corrected = {}
    for key in corr_values.keys():
        corr_values_mean_corrected[key] = np.mean((corr_values[key] - across_model_average + grand_average),axis=1)
        print(np.mean(corr_values_mean_corrected[key]))
    
    corr = []
    net = []
    for key in corr_values_mean_corrected.keys():
        mean_across_crossval = np.mean(corr_values[key],axis=1)
        corr.extend(mean_across_crossval)
        net.extend(np.repeat(key,len(mean_across_crossval)))
        #corr.extend(corr_values_mean_corrected[key])
        #net.extend(np.repeat(key,len(corr_values_mean_corrected[key])))
    dict_all = {'corr': corr,
                'net': net}
    df_all = pd.DataFrame(dict_all)
    df_layer2 = df_all[(df_all['net']!= 'all-alexnettrained-with-dcrandom3') & (df_all['net']!= 'all-dctrained-with-dcrandom3')]
    df_layer3 = df_all[(df_all['net']!= 'all-alexnettrained-with-dcrandom2') & (df_all['net']!= 'all-dctrained-with-dcrandom2')]

    errbars_layer2 = []
    errbars_layer3 = []

    for net in df_layer2['net'].unique():
        errbars_layer2.append(1.96*sem(df_layer2[df_layer2['net']==net]['corr']))

    for net in df_layer3['net'].unique():
        errbars_layer3.append(1.96*sem(df_layer3[df_layer3['net']==net]['corr']))

    g1=sns.violinplot(x='net',y='corr',data=df_layer2, ci=None)
    g1.set_xticklabels(g1.get_xticklabels(), size = 5)
    #x_coords = [p.get_x() + 0.5*p.get_width() for p in g1.patches]
    #y_coords = [p.get_height() for p in g1.patches]
    #plt.errorbar(x_coords, y_coords, fmt='none', yerr=errbars_layer2, c="black", elinewidth=2)
    plt.suptitle('Addition of layer 2 from random dc')
    plt.ylim((0,0.4))
    g1.text(0, 0.30, "a", ha='center', va='bottom',fontsize = 10)  
    g1.text(1, 0.30, "ab", ha='center', va='bottom',fontsize = 10)  
    g1.text(2, 0.35, "ab", ha='center', va='bottom',fontsize = 10)  
    g1.text(3, 0.35, "b", ha='center', va='bottom',fontsize = 10)  
    g1.text(4, 0.35, "b", ha='center', va='bottom',fontsize = 10)  
    plt.savefig('/home/annatruzzi/multiple_deepcluster/figures/bootstrapping_plots_withdcrandomd_layer2.png')
    plt.close()

    g2=sns.violinplot(x='net',y='corr',data=df_layer2, ci=None)
    g2.set_xticklabels(g2.get_xticklabels(), size = 5)
    plt.ylim((0,0.4))
    #x_coords = [p.get_x() + 0.5*p.get_width() for p in g2.patches]
    #y_coords = [p.get_height() for p in g2.patches]
    #plt.errorbar(x_coords, y_coords, fmt='none', yerr=errbars_layer3, c="black", elinewidth=2)
    g2.text(0, 0.30, "a", ha='center', va='bottom',fontsize = 10)  
    g2.text(1, 0.30, "ab", ha='center', va='bottom',fontsize = 10)  
    g2.text(2, 0.35, "ab", ha='center', va='bottom',fontsize = 10)  
    g2.text(3, 0.35, "b", ha='center', va='bottom',fontsize = 10)  
    g2.text(4, 0.35, "ab", ha='center', va='bottom',fontsize = 10)  
    plt.suptitle('Addition of layer 3 from random dc')
    plt.savefig('/home/annatruzzi/multiple_deepcluster/figures/bootstrapping_plots_withdcrandom_layer3.png')
    plt.close()
    a=1

if __name__ == '__main__':
    main()