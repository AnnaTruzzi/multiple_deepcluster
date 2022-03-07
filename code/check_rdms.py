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

layers = ['ReLu1', 'ReLu2', 'ReLu3', 'ReLu4', 'ReLu5','ReLu6','ReLu7']
states = ['randomstate','100epochs']
instances = 11
list_file = []

for instance in range(1,instances):
    for layer in layers:
        list_file.append(f'dc{instance}_100epochs_{layer}')
        list_file.append(f'dc{instance}_randomstate_{layer}')

for instance in range(1,instances):
    for state in states:
        for layer in layers:
            curr_list = [i for i in list_file if state in i and layer in i]
            all_combinations = combinations(curr_list,2)
            for comb in all_combinations:
                with open((f'/data/multiple_deepcluster/rdms/{comb[0]}.pickle'), 'rb') as f:
                    rdm1 = pickle.load(f)
                with open((f'/data/multiple_deepcluster/rdms/{comb[1]}.pickle'), 'rb') as f:
                    rdm2 = pickle.load(f)
                diff = rdm1 - rdm2
                if np.all((diff == 0)):
                    print(f'******* {comb[0]} and {comb[1]} ARE IDENTICAL!!**********')
                else:
                    print(f'{comb[0]} and {comb[1]} are different')
