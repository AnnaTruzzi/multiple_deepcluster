import os
import scipy.io
import pickle
import numpy as np
from itertools import compress
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
import statsmodels.formula.api as smf
from PIL import Image


## load activations dictionary 
def load_dict(path):
    with open(path,'rb') as file:
        dict=pickle.load(file, encoding="latin1")
    return dict

   
def loadmat(matfile):
    try:
        f = h5py.File(matfile)
    except (IOError, OSError):
        return io.loadmat(matfile)
    else:
        return {name: np.transpose(f.get(name)) for name in f.keys()}

def reorder_od(dict1,order):
   new_od = collections.OrderedDict([(k,None) for k in order if k in dict1])
   new_od.update(dict1)
   return new_od

def presentation_order(ordered_names_dict):
    orderedNames = []
    for item in sorted(ordered_names_dict.items()):
        orderedNames.append(item[1])
    return orderedNames

def get_visual_features(img_pil,backcol=None):
  img=np.asarray(img_pil).astype('double')
  lum=np.mean(img,axis=2)
  hsv=np.asarray(img_pil.convert('HSV'))
  
  # If background colour not provided, get it from first pixel (may be risky!) number of pixels
  if not backcol:  
    backcol=img[0,0]
    print('Using top left pixel as background colour, value %d %d %d'%(backcol[0],backcol[1],backcol[2]))
  backlum=np.mean(backcol)

  # All pixels background colour
  foreground=np.any(img!=backcol,2)
  
  img_fg=img[foreground]
  hsv_fg=np.asarray(hsv)[foreground]
  
  features={}
  # Size - total of non-background pixels
  features['size']=np.sum(foreground)
  # Contrast - mean of sum of squared difference of each colour channel from background colour
  features['contrast']=np.mean(np.sum(np.power(img_fg-backcol,2.0),1))
  # Hue - mean of H
  features['hue']=np.mean(hsv_fg[:,0])
  # Lurid - mean standard deviation across colour channel
  features['lurid']=np.mean(np.std(img_fg,1))
  # Thinness 
  y,x=np.where(foreground)
  y=y-np.mean(y)
  x=x-np.mean(x)
  a=np.vstack((x,y))
  u,s,v=np.linalg.svd(a,full_matrices=False)
  features['thinness']=s[0]/s[1]
  # Radians away from being horizontal
  features['radiansoffhorizontal']=np.abs(np.arctan(u[0,1]/u[0,0]))
  
  return features


def main():
    dcrandom = {}
    dctrained = {}
    alexnettrained = {}

    for layer in layers:
        dcrandom_layer_rdms = []
        dctrained_layer_rdms = []
        for instance in range(1, dc_instances):
            with open((f'/data/multiple_deepcluster/rdms/dc{instance}_randomstate_{layer}.pickle'), 'rb') as handle:
                dcrandom_rdm = pickle.load(handle)
            with open((f'/data/multiple_deepcluster/rdms/dc{instance}_100epochs_{layer}.pickle'), 'rb') as handle:
                dctrained_rdm = pickle.load(handle)
            dcrandom_layer_rdms.append(dcrandom_rdm)
            dctrained_layer_rdms.append(dctrained_rdm)
        dcrandom_mean = squareform((np.mean(np.array(dcrandom_layer_rdms),axis=0)))
        dcrandom[f'dcrandom_{layer}'] = dcrandom_mean - np.mean(dcrandom_mean)
        
        dctrained_mean = squareform((np.mean(np.array(dctrained_layer_rdms),axis=0)))        
        dctrained[f'dctrained_{layer}'] = dctrained_mean - np.mean(dctrained_mean)
        with open((f'/data/multiple_deepcluster/rdms/alexnet1_pretrained_{layer}.pickle'), 'rb') as handle:
            alexnettrained_mean = squareform(pickle.load(handle))
            alexnettrained[f'alexnettrained_{layer}'] = alexnettrained_mean - np.mean(alexnettrained_mean)

    all_nets = {'dcrandom': dcrandom,
                'dctrained': dctrained,
                'alexnettrained': alexnettrained}
    
    fmri_pth = '/data/algonautsChallenge2019/Training_Data/92_Image_Set/target_fmri.mat'
    fmri_mat = loadmat(fmri_pth)
    IT = fmri_mat['IT_RDMs']
    IT_mean = squareform(np.mean(IT,axis = 0),checks=False)
    IT_mean_corrected = IT_mean - np.mean(IT_mean)

    EVC = fmri_mat['EVC_RDMs']
    EVC_mean = squareform(np.mean(EVC,axis = 0),checks=False)
    EVC_mean_corrected = EVC_mean - np.mean(EVC_mean)

    all_ROIs = {'IT_mean_corrected': IT_mean_corrected,
                'EVC_mean_corrected':EVC_mean_corrected}

    ############  Semantic features
    semantic_features_vec = np.array([1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,
                3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6])

    animacy=np.array([1]*48+[2]*44) # Am I right there are 48 animate things?

    ############  Perceptual features
    #perceptual_features_dic = collections.OrderedDict()
    perceptual_features_dic = {'size': collections.OrderedDict(), 
                            'contrast': collections.OrderedDict(), 
                            'hue': collections.OrderedDict(), 
                            'lurid': collections.OrderedDict(), 
                            'thinness': collections.OrderedDict(), 
                            'radiansoffhorizontal': collections.OrderedDict()}

    allimgs=[]
    allimg_names=[]
    backcol=[255,255,255]
    for img in glob.glob(os.path.join('/data/algonautsChallenge2019/Training_Data/92_Image_Set/92images/jpg_images','*.jpg')):
        print(img)
        img_loaded = Image.open(img)
        features = get_visual_features(img_loaded,backcol)
        img_name = img.split('/')[-1].split('.')[0].split('_')[-1] 
        perceptual_features_dic['size'][img_name] = features['size']
        perceptual_features_dic['contrast'][img_name] = features['contrast']
        perceptual_features_dic['hue'][img_name] = features['hue']
        perceptual_features_dic['lurid'][img_name] = features['lurid']
        perceptual_features_dic['thinness'][img_name] = features['thinness']
        perceptual_features_dic['radiansoffhorizontal'][img_name] = features['radiansoffhorizontal']

        # Prepare pixel overlap measure
        img=np.asarray(img_loaded).astype('double')
        # All pixels background colour
        allimgs.append(np.any(img!=backcol,2))
        allimg_names.append(img_name)

    ind=np.argsort(allimg_names)
    allimgs=np.array(allimgs)
    allimgs=allimgs[ind,:]
    allimgs=np.reshape(allimgs,(allimgs.shape[0],-1))
    silhouetterdm=np.corrcoef(allimgs)    

    perceptual_features_dic['size'] = reorder_od(perceptual_features_dic['size'], sorted(perceptual_features_dic['size'].keys()))
    perceptual_features_dic['contrast'] = reorder_od(perceptual_features_dic['contrast'], sorted(perceptual_features_dic['contrast'].keys()))
    perceptual_features_dic['hue'] = reorder_od(perceptual_features_dic['hue'], sorted(perceptual_features_dic['hue'].keys()))
    perceptual_features_dic['lurid'] = reorder_od(perceptual_features_dic['lurid'], sorted(perceptual_features_dic['lurid'].keys()))
    perceptual_features_dic['thinness'] = reorder_od(perceptual_features_dic['thinness'], sorted(perceptual_features_dic['thinness'].keys()))
    perceptual_features_dic['radiansoffhorizontal'] = reorder_od(perceptual_features_dic['radiansoffhorizontal'], sorted(perceptual_features_dic['radiansoffhorizontal'].keys()))

    # Refactored
    sem_feats=['semantic_category', 'semantic_animacy']
    perc_feats=['size','contrast','hue','lurid','thinness','radiansoffhorizontal','silhouette']
    all_feats=sem_feats+perc_feats

    feat_rdms={}
    a=np.tile(semantic_features_vec,[92,1])
    feat_rdms['semantic_category']=1-(a==a.T)
    a=np.tile(animacy,[92,1])
    feat_rdms['semantic_animacy']=1-(a==a.T)

    # Refactored.
    for perc_feat in perc_feats:
        if perc_feat=='silhouette':
            feat_rdms[perc_feat]=silhouetterdm
        else:
            pf=np.array([perceptual_features_dic[perc_feat][x] for x in perceptual_features_dic[perc_feat]])
            a=np.tile(pf,[92,1])
            feat_rdms[perc_feat]=np.abs(a-a.T) # Euclidian distance between two scalars is just their absolute difference

    # Squareform and zero centre
    for k in feat_rdms:
        feat_rdms[k]=squareform(feat_rdms[k],checks=False)
        feat_rdms[k]=feat_rdms[k]-np.mean(feat_rdms[k])


    test_ROIs=['IT','EVC']
    test_nets=['dcrandom','dctrained','alexnettrained']
    test_layers=['ReLu2','ReLu7']
    proportion_list=[]
    type_list=[]
    ROI_list=[]
    net_list=[]
    out_layer_list=[]

    for test_ROI in test_ROIs:
        for test_net in test_nets:
            with open(f'/home/annatruzzi/multiple_deepcluster/results/mediation_results_{test_ROI}_{test_net}.txt', 'w') as f:
                f.write('=' * 20 + '\n')
                f.write(f'{test_net} VS {test_ROI} \n')
                f.write('=' * 20 + '\n')
                for test_layer in test_layers:
                    f.write(f'Results for {test_layer} \n')
                    print(test_ROI)
                    print(test_net)
                    print(test_layer)
                    dfcols={f'{test_net}': all_nets[f'{test_net}'][f'{test_net}_{test_layer}'], f'{test_ROI}': all_ROIs[f'{test_ROI}_mean_corrected']}
                    for feat in all_feats:
                        dfcols[feat]=feat_rdms[feat]    
                    df=pd.DataFrame(dfcols)
                    res =pg.mediation_analysis(data=df, x=test_net, m=all_feats, y=test_ROI, alpha=0.05, seed=42, return_dist=False)
                    print(res)

                    # Extract key values
                    perc_indirect=np.sum([float(res['coef'][res['path']=='Indirect '+f]) for f in perc_feats])
                    sem_indirect=np.sum([float(res['coef'][res['path']=='Indirect '+f]) for f in sem_feats])
                    tot=float(res['coef'][res['path']=='Total'])
                    direct=float(res['coef'][res['path']=='Direct'])

                    proportion_list.extend([perc_indirect,sem_indirect])
                    type_list.extend(['perceptual','semantic'])
                    ROI_list.extend(np.repeat(test_ROI,2))
                    net_list.extend(np.repeat(test_net,2))
                    out_layer_list.extend(np.repeat(test_layer,2))

                    print('Indirect perceptual %.3f  semantic %.3f'%(perc_indirect, sem_indirect))
                    print('Proportion of total for indirect perceptual %0.3f  semantic %0.3f'%(perc_indirect/tot, sem_indirect/tot))
                    print('Proportion of total for indirect overall %0.3f'%((perc_indirect+sem_indirect)/tot))
                    
                    f.write('\n')
                    f.write(res.to_string(index=False))
                    f.write('\n')
                    f.write('\n' + 'Indirect perceptual %.3f  semantic %.3f'%(perc_indirect, sem_indirect))
                    f.write('\n' + 'Proportion of total for indirect perceptual %0.3f  semantic %0.3f'%(perc_indirect/tot, sem_indirect/tot))
                    f.write('\n' + 'Proportion of total for indirect overall %0.3f'%((perc_indirect+sem_indirect)/tot))
                    f.write('\n')

    out_dict = {'proportion': proportion_list,
                'type': type_list,
                'ROI': ROI_list,
                'net': net_list,
                'layer': out_layer_list}
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv('/home/annatruzzi/multiple_deepcluster/results/mediation_proportion_to_total.csv')

if __name__ == '__main__':
    layers = ['ReLu1', 'ReLu2', 'ReLu3', 'ReLu4', 'ReLu5','ReLu6','ReLu7']
    dc_instances = 11

    img_names_pth = '/home/annatruzzi/multiple_deepcluster/data/niko92_img_names.pickle' 
    img_names = load_dict(img_names_pth)
    img_names = reorder_od(img_names, sorted(img_names.keys()))
    orderedNames = presentation_order(img_names)

    main()
