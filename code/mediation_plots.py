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

def main():

    df = pd.read_csv('/home/annatruzzi/multiple_deepcluster/results/mediation_proportion_to_total_allsubj.csv')

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
            x_coords = [p.get_x() + 0.5*p.get_width() for p in ax[i].patches]
            y_coords = [p.get_height() for p in ax[i].patches]
            plt.errorbar(x_coords, y_coords, fmt='none', yerr=[sem((np.array(df[(df['net']==net) & (df['ROI']==ROI) & (df['type']=='perceptual') & (df['layer']=='ReLu2')]['proportion'])-np.array(df[(df['net']==net) & (df['ROI']==ROI) & (df['type']=='semantic') & (df['layer']=='ReLu2')]['proportion']))/2),
                        sem((np.array(df[(df['net']==net) & (df['ROI']==ROI) & (df['type']=='perceptual') & (df['layer']=='ReLu2')]['proportion'])-np.array(df[(df['net']==net) & (df['ROI']==ROI) & (df['type']=='semantic') & (df['layer']=='ReLu2')]['proportion']))/2),
                        sem((np.array(df[(df['net']==net) & (df['ROI']==ROI) & (df['type']=='perceptual') & (df['layer']=='ReLu2')]['proportion'])-np.array(df[(df['net']==net) & (df['ROI']==ROI) & (df['type']=='semantic') & (df['layer']=='ReLu7')]['proportion']))/2),
                        sem((np.array(df[(df['net']==net) & (df['ROI']==ROI) & (df['type']=='perceptual') & (df['layer']=='ReLu2')]['proportion'])-np.array(df[(df['net']==net) & (df['ROI']==ROI) & (df['type']=='semantic') & (df['layer']=='ReLu7')]['proportion']))/2)], c="black", elinewidth=2)
            ax[i].set(ylim=(0, 1.3))
            if i==0:
                ax[i].set_ylabel('Proportion of total effect explained')
            else:
                ax[i].set_ylabel('')
                ax[i].yaxis.set_ticklabels([])
            ax[i].set_title(f'{net}')
        plt.suptitle(f'{ROI}')
        plt.savefig(f'/home/annatruzzi/multiple_deepcluster/figures/mediation_plots_{ROI}_allsubj.png')
        plt.savefig(f'/home/annatruzzi/multiple_deepcluster/figures/mediation_plots_{ROI}_allsubj.pdf')


if __name__ == '__main__':
    networks = ['dcrandom','dctrained','alexnettrained']
    ROIs = ['EVC', 'IT']
    main()