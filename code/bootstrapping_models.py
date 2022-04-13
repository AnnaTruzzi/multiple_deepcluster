import os
import scipy.io
import pickle
import numpy as np
from itertools import compress
from scipy.spatial import distance
import sklearn
from sklearn.manifold import MDS
from statsmodels.stats.anova import AnovaRM
import matplotlib
from matplotlib import pyplot as plt
import re
from scipy.cluster.hierarchy import dendrogram, linkage
from itertools import combinations
import scipy.io as io
import h5py
import hdf5storage
from scipy import stats
from scipy.spatial.distance import squareform
import collections
import re
import skbio
import seaborn as sns
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
import scipy
from scipy.stats import pearsonr, spearmanr, kendalltau
import statsmodels.formula.api as smf
#import pkg_resources
import pingouin as pg
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
import random
from scipy.optimize import nnls




## load activations dictionary 
def load_dict(path):
    with open(path,'rb') as file:
        dict=pickle.load(file, encoding="latin1")
    return dict

def reorder_od(dict1,order):
   new_od = collections.OrderedDict([(k,None) for k in order if k in dict1])
   new_od.update(dict1)
   return new_od
   
def loadmat(matfile):
    try:
        f = h5py.File(matfile)
    except (IOError, OSError):
        return io.loadmat(matfile)
    else:
        return {name: np.transpose(f.get(name)) for name in f.keys()}

def presentation_order(ordered_names_dict):
    orderedNames = []
    for item in sorted(ordered_names_dict.items()):
        orderedNames.append(item[1])
    return orderedNames

def rdm(act_matrix):
    rdm_matrix = distance.squareform(distance.pdist(act_matrix,metric='correlation'))
    return rdm_matrix

# calculate aic for regression
def calculate_aic(n, mse, num_params):
	aic = n * log(mse) + 2 * num_params
	return aic
    

def main():
#    githubpth='/home/CUSACKLAB/annatruzzi/cichy2016'
#    basepth='/home/rhodricusack/Documents/truzzi_unsupervised_brain'
#    githubpth=os.path.join(basepth,'unsupervised_brain')
#    algonautspth=os.path.join(basepth,'Images_MRI') # 'algonautsChallenge2019/Training_Data/92_Image_Set'

    layers = ['ReLu1', 'ReLu2', 'ReLu3', 'ReLu4', 'ReLu5','ReLu6','ReLu7']
#    dcalexnet_random_pth = githubpth + '/niko92_activations_untrained_DC.pickle'
#    standardalexnet_trained_pth = githubpth + '/niko92_activations_pretrained_torchalexnetINonly.pickle'
#    img_names_pth = githubpth + '/niko92_img_names.pickle'
#    img_pth = algonautspth + '92images/jpg_images/'

    dcalexnet_random_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/niko92_activations_untrained_DC.pickle'
    standardalexnet_trained_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/niko92_activations_pretrained_torchalexnetINonly.pickle'
    standardalexnet_random_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/niko92_activations_untrained_torchalexnet.pickle'
    img_names_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/niko92_img_names.pickle'
    img_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/algonautsChallenge2019/Training_Data/92_Image_Set/92images/jpg_images/'
 
    dcalexnet_random = load_dict(dcalexnet_random_pth)
    standardalexnet_trained = load_dict(standardalexnet_trained_pth)
    standardalexnet_random = load_dict(standardalexnet_random_pth)
    img_names = load_dict(img_names_pth)
    img_names = reorder_od(img_names, sorted(img_names.keys()))

    orderedNames = presentation_order(img_names)
    orderedKeys_dcalexnet_random = []
    orderedKeys_standardalexnet_trained = []
    orderedKeys_standardalexnet_random = []
    for name in orderedNames:
        number = [i[0] for i in img_names.items() if i[1] == name]
        key_dcalexnet_random = [k for k in dcalexnet_random.keys() if '%s.jpg' %number[0] in k]
        key_standardalexnet_trained = [k for k in standardalexnet_trained.keys() if '%s.jpg' %number[0] in k]
        key_standardalexnet_random = [k for k in standardalexnet_random.keys() if '%s.jpg' %number[0] in k]
        orderedKeys_dcalexnet_random.append(key_dcalexnet_random[0])
        orderedKeys_standardalexnet_trained.append(key_standardalexnet_trained[0])
        orderedKeys_standardalexnet_random.append(key_standardalexnet_random[0])
    
    ############ order activations dictionary and reorganize it in order to get one dictionary per layer and image (instead of one dictionary per image)
    ordered_dcalexnet_random = reorder_od(dcalexnet_random,orderedKeys_dcalexnet_random)
    ordered_standardalexnet_trained = reorder_od(standardalexnet_trained,orderedKeys_standardalexnet_trained)
    ordered_standardalexnet_random = reorder_od(standardalexnet_random,orderedKeys_standardalexnet_random)

    layer_dcalexnet_random = collections.OrderedDict()
    for layer in range(0,len(layers)):
        layer_dcalexnet_random[layer] = collections.OrderedDict()
        for item in ordered_dcalexnet_random.items():
                if layer < 5:
                    layer_dcalexnet_random[layer][item[0]] = np.mean(np.mean(np.squeeze(item[1][layer]),axis = 2), axis = 1)
                else:
                    layer_dcalexnet_random[layer][item[0]] = np.squeeze(item[1][layer])    

    layer_standardalexnet_trained = collections.OrderedDict()
    for layer in range(0,len(layers)):
        layer_standardalexnet_trained[layer] = collections.OrderedDict()
        for item in ordered_standardalexnet_trained.items():
                if layer < 5:
                    layer_standardalexnet_trained[layer][item[0]] = np.mean(np.mean(np.squeeze(item[1][layer]),axis = 2), axis = 1)
                else:
                    layer_standardalexnet_trained[layer][item[0]] = np.squeeze(item[1][layer])    

    layer_standardalexnet_random = collections.OrderedDict()
    for layer in range(0,len(layers)):
        layer_standardalexnet_random[layer] = collections.OrderedDict()
        for item in ordered_standardalexnet_random.items():
                if layer < 5:
                    layer_standardalexnet_random[layer][item[0]] = np.mean(np.mean(np.squeeze(item[1][layer]),axis = 2), axis = 1)
                else:
                    layer_standardalexnet_random[layer][item[0]] = np.squeeze(item[1][layer])    

    ####### transform layer activations in matrix and calculate rdms
    rdm_dcalexnet_random = []
    for l in layer_dcalexnet_random.keys():
        curr_layer = np.array([layer_dcalexnet_random[l][i] for i in orderedKeys_dcalexnet_random])
        rdm_curr_layer = rdm(curr_layer)
        corrected_rdm = squareform(squareform(rdm_curr_layer) - np.mean(squareform(rdm_curr_layer)))
        rdm_dcalexnet_random.append(rdm(corrected_rdm))
    
    rdm_standardalexnet_trained = []
    for l in layer_standardalexnet_trained.keys():
        curr_layer = np.array([layer_standardalexnet_trained[l][i] for i in orderedKeys_standardalexnet_trained])
        rdm_curr_layer = rdm(curr_layer)
        corrected_rdm = squareform(squareform(rdm_curr_layer) - np.mean(squareform(rdm_curr_layer)))
        rdm_standardalexnet_trained.append(corrected_rdm)

    rdm_standardalexnet_random = []
    for l in layer_standardalexnet_random.keys():
        curr_layer = np.array([layer_standardalexnet_random[l][i] for i in orderedKeys_standardalexnet_random])
        rdm_curr_layer = rdm(curr_layer)
        corrected_rdm = squareform(squareform(rdm_curr_layer) - np.mean(squareform(rdm_curr_layer)))
        rdm_standardalexnet_random.append(corrected_rdm)

    ###### load fmri data
#    fmri_pth = os.path.join(algonautspth, 'target_fmri.mat')

    fmri_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/algonautsChallenge2019/Training_Data/92_Image_Set/target_fmri.mat'
    fmri_mat = loadmat(fmri_pth)
    IT = fmri_mat['IT_RDMs']
    IT_mean = squareform(squareform(np.mean(IT,axis = 0),checks=False))
    IT_mean_corrected = squareform(squareform(IT_mean) - np.mean(squareform(IT_mean)))


    ###############  Linear regression leave one subject out with cross validate
    alexnet_dict={'standardalexnet_trained_%d'%x : rdm_standardalexnet_trained[x] for x in range(7)}
    
    dc_dict=dict(alexnet_dict)
    dc_dict['deepcluster_layer_2']=rdm_dcalexnet_random[1]
    
    alexnet7_dc2_dict = {'deepcluster_layer_2': rdm_dcalexnet_random[1],
                        'standardalexnet_trained_6': rdm_standardalexnet_trained[6]}
    
    alexnet_random_dict ={'standardalexnet_trained_%d'%x : rdm_standardalexnet_random[x] for x in range(7)}
    
    alexnetall_with_alexnetrandom2_dict=dict(alexnet_dict)
    alexnetall_with_alexnetrandom2_dict['randomalexnet_layer_2']=rdm_standardalexnet_random[1]
    
    # Run through two models - one without dc2 and one with
    model_list=[{'name':'alexnet-all-layers', 'dict':alexnet_dict}, 
                {'name':'alexnet-all-layers_with_dc2', 'dict':dc_dict},
                {'name':'alexnet-layer-7_with_dc2', 'dict':alexnet7_dc2_dict},
                {'name':'all-random-alexnet','dict':alexnet_random_dict},
                {'name':'alexnetall_with_alexnetrandom2','dict':alexnetall_with_alexnetrandom2_dict}]                

    #model_list=[{'name':'alexnet7_with_alexnetrandom2','dict':alexnet7_with_alexnetrandom2_dict}]                

    corr_kt={}
    out_corr_model={}

    with open('/home/CUSACKLAB/annatruzzi/unsupervised_brain/CombinedRegressionModels_results_NNLS.txt', 'w') as f:
        for model in model_list:
            print('***' + model['name'])
            f.write('\n' + '#' * 20 + '\n')
            f.write('***' + model['name'] + '\n')

            # Holds order consistent
            layers=list(model['dict'].keys())
            
            # DNN activations, array of shape [squareform RDM size,layers] 
            X=np.array([squareform(model['dict'][k]) for k in layers]).T

            # fMRI activity, array of shape [subjects,squareform RDM size] 
            Y = []
            for ITonesubj in IT:
                Y.append(squareform(ITonesubj,checks = False))
            Y=np.array(Y).T
            
            # Mean centre 
            X=X-np.mean(X,0)[np.newaxis,:]
            Y=Y-np.mean(Y,0)[np.newaxis,:]

            # Set up linear regression 
            regr=sklearn.linear_model.LinearRegression()
        
            # Leave-one-subject-out cross validation     
            kfold = sklearn.model_selection.KFold(n_splits=15, shuffle=False)
            corr_kt[model['name']]=[]
            coef = []
            for train_index,test_index in kfold.split(Y.T):
                # Fit model to N-1 subjects
                regr.fit(X, np.mean(Y[:,train_index],axis=1))
                coef.append(regr.coef_)
                # Get prediction for RDM
                predicted_rdm=regr.predict(X)
                # Test prediction in test subject Kendall's Tau
                corr,pvalue=kendalltau(predicted_rdm,Y[:,test_index])
                corr_kt[model['name']].append(corr)    
        #       Other options for calculating r-squared:
        #        corr_kt.append(regr.score(X,Y[:,test_index]))        
        #        corr_kt.append(sklearn.metrics.r2_score(Y[:,test_index],predicted_rdm))        

            # Is prediction above chance?
            # Note that if DNN models were random this probably wouldn't be zero
            t,prob=scipy.stats.ttest_1samp(corr_kt[model['name']], popmean=0)

            betas = np.mean(np.array(coef),axis = 0)

            print('Cross validated Kendall''s Tau mean %f +/- se %f'%(np.mean(corr_kt[model['name']]),scipy.stats.sem(corr_kt[model['name']])))
            print(' t=%f p<%f'%(t,prob))
            f.write('* LeaveOneOut' + '\n')
            f.write('Cross validated Kendall''s Tau mean %f +/- se %f'%(np.mean(corr_kt[model['name']]),scipy.stats.sem(corr_kt[model['name']])) + '\n')
            f.write('t=%f p<%f'%(t,prob) + '\n')
            f.write('\n')

            '''# Let's test individual coefficients a different way 
            regr.fit(X,Y)
            coef=regr.coef_
            # Get all subjects coefficients from each layer
            
            for layerind, column in enumerate(coef.T):
                # Test this coefficient
                t,prob=scipy.stats.ttest_1samp(column, popmean=0)
                print('Layer %s t=%f, p<%f'%(layers[layerind],t,prob))'''

        
            np.random.seed(1234)
            # Storrs method
            nbootstraps=20000
            ncrossvals=20
            nsubj=IT.shape[0]
            nstim=IT.shape[1]
            ntestsubj=4
            nteststim=12
            corr_model=np.zeros((nbootstraps,ncrossvals))   # model
            corr_nl=np.zeros((nbootstraps,ncrossvals))    # noise upper
            corr_nu=np.zeros((nbootstraps,ncrossvals))    # noise lower
            for bootstrap in range(nbootstraps):
                print(bootstrap)
                # Bootstrap subjects and stimuli by resampling with replacement
                subject_bootstrap=np.random.choice(nsubj, nsubj, replace=True)
                stim_bootstrap=np.random.choice(nstim, nstim, replace=True)
                
                # For noise ceiling - upper bound, average of all subjects in this bootstrap 
                allsubjITmean=np.mean(IT[subject_bootstrap,:,:], axis=0)

                for crossval in range(ncrossvals):
                    print(crossval)
                    # Pick test subjects and stimuli from bootstrap by resampling without replacement
                    testsubj=np.random.choice(subject_bootstrap, ntestsubj, replace=False)
                    teststim=np.random.choice(stim_bootstrap, nteststim, replace=False)
                    # Bootstrap samples from other subjects/stimuli are for training
                    trainsubj=[x for x in subject_bootstrap if x not in testsubj]
                    trainstim=[x for x in stim_bootstrap if x not in teststim]
                    # Mean train RDM
                    trainITmean=squareform(np.mean(IT[trainsubj,:,:][:,trainstim,:][:,:,trainstim],0).squeeze(), checks=False)
                    # Get DNN layer rdms
                    trainX=np.array([squareform(model['dict'][k][trainstim,:][:,trainstim]) for k in layers]).T
                    # Mean centre 
                    trainX=trainX-np.mean(trainX,0)[np.newaxis,:]
                    trainITmean=trainITmean-np.mean(trainITmean)
                    
                    '''# Fit layer RDMs to IT (second level fitting in Storrs)
                    regr=sklearn.linear_model.LinearRegression()
                    regr.fit(trainX,trainITmean)
                    # Predict for other subjects and stimuli. 
                    #  Get this weighted model
                    testX=np.array([squareform(model['dict'][k][teststim,:][:,teststim]) for k in layers])
                    testXweightedmean=np.mean(regr.coef_[:,np.newaxis]*testX,0)'''

                    # NNLS regression (as suggested by reviewer at SVRHM)
                    regr_coef,rnorm = nnls(trainX,trainITmean)
                    testX=np.array([squareform(model['dict'][k][teststim,:][:,teststim]) for k in layers])
                    testXweightedmean=np.mean(regr_coef[:,np.newaxis]*testX,0)
                    #  Crossvalidation is done for individual test subjects not the mean.
                    testIT=IT[testsubj,:,:][:,teststim,:][:,:,teststim]
                    # For noise ceiling - lower bound, average of train subjects 
                    trainITmean=squareform(np.mean(IT[trainsubj,:,:][:,teststim,:][:,:,teststim],axis=0),checks=False)
                    # For noise ceiling - upper bound, average of all subjects
                    allsubjITmean_teststim=squareform(allsubjITmean[teststim,:][:,teststim],checks=False)
                    allcorr=[]
                    allnl=[]
                    allnu=[]
                    for ITonesubj in testIT:
                        # One subject at a time
                        Y=squareform(ITonesubj,checks = False)
                        # Evaluate prediction
                        corr,pvalue=kendalltau(testXweightedmean,Y)
                        allcorr.append(corr)
                        # Noise ceiling, lower
                        corr,pvalue=kendalltau(trainITmean,Y)
                        allnl.append(corr)
                        # Noise ceiling, upper
                        corr,pvalue=kendalltau(allsubjITmean_teststim,Y)
                        allnu.append(corr)

                    corr_model[bootstrap,crossval]=np.mean(allcorr)
                    corr_nl[bootstrap,crossval]=np.mean(allnl)
                    corr_nu[bootstrap,crossval]=np.mean(allnu)

            # Average across cross validations
            out_corr_model[model['name']] = corr_model
            corr_model_xvalmean=np.mean(corr_model, axis=1) 
            corr_nl_xvalmean=np.mean(corr_nl, axis=1) 
            corr_nu_xvalmean=np.mean(corr_nu, axis=1) 

            # Display
            print('Cross validated across subjects and items')
            print(' Model mean          %f +/- %f'%(np.mean(corr_model_xvalmean),scipy.stats.sem(corr_model_xvalmean)))
            print(' Lower noise ceiling %f +/- %f'%(np.mean(corr_nl_xvalmean),scipy.stats.sem(corr_nl_xvalmean)))
            print(' Upper noise ceiling %f +/- %f'%(np.mean(corr_nu_xvalmean),scipy.stats.sem(corr_nu_xvalmean)))
            
            f.write('* Cross validated across subjects and items' + '\n')
            f.write('Model mean          %f +/- %f'%(np.mean(corr_model_xvalmean),scipy.stats.sem(corr_model_xvalmean)) + '\n')
            f.write(' Lower noise ceiling %f +/- %f'%(np.mean(corr_nl_xvalmean),scipy.stats.sem(corr_nl_xvalmean)) + '\n')
            f.write(' Upper noise ceiling %f +/- %f'%(np.mean(corr_nu_xvalmean),scipy.stats.sem(corr_nu_xvalmean)) + '\n')
            f.write('\n')

        with open('/home/CUSACKLAB/annatruzzi/unsupervised_brain/CombinedRegressionModels_BootstrappingValues_NNLS.pickle', 'wb') as handle:
            pickle.dump(out_corr_model, handle)

        #with open('/home/CUSACKLAB/annatruzzi/unsupervised_brain/CombinedRegressionModels_BootstrappingValues_randomalexnetonly.pickle', 'wb') as handle:
            #pickle.dump(out_corr_model, handle)

        #with open('/home/CUSACKLAB/annatruzzi/unsupervised_brain/CombinedRegressionModels_BootstrappingValues_allalexnet_with_randomalexnet2.pickle', 'wb') as handle:
            #pickle.dump(out_corr_model, handle)

if __name__ == '__main__':
    main()