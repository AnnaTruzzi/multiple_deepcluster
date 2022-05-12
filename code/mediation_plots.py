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

def main():

    df = pd.read_csv('/home/annatruzzi/multiple_deepcluster/results/mediation_proportion_to_total.csv')

    costum_palette = [sns.xkcd_rgb['old rose'],sns.xkcd_rgb['seafoam blue']]
    sns.set()
    sns.set_palette(costum_palette)
    sns.set_style("white")
    #sns.set_context('paper')
    for ROI in ROIs:
        fig, ax = plt.subplots(nrows=1, ncols=3)
        for i,net in enumerate(networks):
            sns.barplot(x="layer", y="proportion", hue = 'type',
            data=df[(df['net']==net) & (df['ROI']==ROI)], ax = ax[i], ci=None)
            ax[i].set(ylim=(0, 1))
            if i==0:
                ax[i].set_ylabel('Proportion of total effect explained')
            else:
                ax[i].set_ylabel('')
                ax[i].yaxis.set_ticklabels([])
            ax[i].set_title(f'{net}')
        plt.suptitle(f'{ROI}')
        plt.savefig(f'/home/annatruzzi/multiple_deepcluster/figures/mediation_plots_{ROI}.png')


if __name__ == '__main__':
    networks = ['dcrandom','dctrained','alexnettrained']
    ROIs = ['EVC', 'IT']
    main()