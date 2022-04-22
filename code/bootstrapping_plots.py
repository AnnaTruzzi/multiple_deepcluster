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
    corr = []
    net = []
    for key in corr_values.keys():
        mean_across_crossval = np.mean(corr_values[key],axis=1)
        corr.extend(mean_across_crossval)
        net.extend(np.repeat(key,len(mean_across_crossval)))
    dict_all = {'corr': corr,
                'net': net}
    df_all = pd.DataFrame(dict_all)
    df_layer2 = df_all[(df_all['net']!= 'all-alexnettrained-with-dcrandom3') & (df_all['net']!= 'all-dctrained-with-dcrandom3')]
    df_layer3 = df_all[(df_all['net']!= 'all-alexnettrained-with-dcrandom2') & (df_all['net']!= 'all-dctrained-with-dcrandom2')]

    g1=sns.violinplot(x='net',y='corr',data=df_layer2)
    g1.set_xticklabels(g1.get_xticklabels(), size = 5)
    plt.suptitle('Addition of layer 2 from random dc')
    plt.savefig('/home/annatruzzi/multiple_deepcluster/figures/bootstrapping_plots_withdcrandomd_layer2.png')

    g2=sns.violinplot(x='net',y='corr',data=df_layer2)
    g2.set_xticklabels(g2.get_xticklabels(), size = 5)
    plt.suptitle('Addition of layer 3 from random dc')
    plt.savefig('/home/annatruzzi/multiple_deepcluster/figures/bootstrapping_plots_withdcrandom_layer3.png')
    a=1

if __name__ == '__main__':
    main()