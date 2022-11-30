import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import awkward as ak
import uproot
from tqdm.auto import tqdm

num_of_tracks = 10

# define the features that we want to extract
features = ['trackBTag_DeltaR',
            'trackBTag_Eta',
            'trackBTag_EtaRel',
            'trackBTag_JetDistVal',
            'trackBTag_Momentum',
            'trackBTag_PPar',
            'trackBTag_PParRatio',
            'trackBTag_PtRatio',
            'trackBTag_PtRel',
            'trackBTag_Sip2dSig',
            'trackBTag_Sip2dVal',
            'trackBTag_Sip3dSig',
            'trackBTag_Sip3dVal',
            'track_charge',
            'track_deltaR',
            'track_drminsv',
            'track_drsubjet1',
            'track_drsubjet2',
            'track_dxy',
            'track_dxysig',
            'track_dz',
            'track_dzsig',
            'track_erel',
            'track_etarel',
            'track_mass',
            'track_phirel',
            'track_pt',
            'track_ptrel']


def get_labels(tree,label):
    '''
    Function to return the labels array out of a root tree.
    This function is required because we use 2 sets of 2 lables each
    where each set is combined with a logical "and".
    For instance "fj_isQCD and sample_isQCD" is the final label to label a jet as
    originating from QCD.
    '''
    prods = label.split('*')
    facts = tree.arrays(prods,library='np')
    labels = np.multiply(facts[prods[0]],facts[prods[1]])
    return labels

def return_sorted_3D_array_awkward(path, num_of_tracks = 10, sort_by = 'trackBTag_Momentum'):
    
    file = uproot.open(path)
    events = file['deepntuplizer/tree']
    max_length  = np.max(events['n_tracks'])
    arrays = events.arrays(features, library='ak')
    momenta = arrays['trackBTag_Momentum']
    padded_momenta = ak.fill_none(ak.pad_none(momenta,max_length,clip=True),0)
    indices = ak.argsort(padded_momenta,axis=1,ascending=False)
    
    sorted_arrays = []

    for key in arrays.fields:
        padded = ak.fill_none(ak.pad_none(arrays[key],max_length,clip=True),0)
        sorted_arrays.append(padded[indices][:,:10].to_numpy())
    
    stacked_3d_array = np.stack(sorted_arrays,axis=1)
    label_array = np.stack([get_labels(events,i) for i in labels],axis=-1)
    
    return stacked_3d_array, label_array

def get_final_arrays(feature_array, label_array):
    
    signals = feature_array[label_array[:,1]==1]
    backgrounds = feature_array[label_array[:,0]==1]
    sample_idx = np.random.choice(np.arange(len(backgrounds)),size=len(signals), replace=False)
    backgrounds = backgrounds[sample_idx]
    hbb = np.concatenate((np.ones(len(signals)),np.zeros(len(backgrounds))), axis=None)
    QCD = np.concatenate((np.zeros(len(signals)),np.ones(len(backgrounds))), axis=None)
    X_train = np.vstack((signals,backgrounds))
    y_train = np.stack((hbb,QCD),axis=1)
    shuffle_idx = np.arange(len(X_train))
    np.random.shuffle(shuffle_idx)
    X_train_final = X_train[shuffle_idx]
    y_train_final = y_train[shuffle_idx]
    
    return X_train_final, y_train_final


def main():

    start = int(sys.argv[1])
    end = int(sys.argv[2]) + 1
    outfile = sys.argv[3]


    final_features = np.empty((0,nfeatures,num_of_tracks))
    final_labels = np.empty((0,nlabels))

    for i in tqdm(range(start, end)):
        file = f"ntuple_merged_{i}.root"
        try:
            feature_array_all, label_array_all = return_sorted_3D_array_awkward(f'../root_files/{file}')
            feature_array, label_array = get_final_arrays(feature_array_all, label_array_all)
            final_features = np.vstack((final_features, feature_array))
            final_labels = np.vstack((final_labels, label_array))    
        except FileNotFoundError:
            pass
        
    print(final_features.shape)
    print(final_labels.shape)

    np.savez(f'../root_files/{outfile}.npz', features = final_features, labels = final_labels, names = features)

if __name__ == '__main__':

    # 2 labels: QCD or Hbb. Logical "and" of labels is used.
    labels = ['fj_isQCD*sample_isQCD',
              'fj_isH*fj_isBB']

    nfeatures = len(features)
    nlabels = len(labels)
    main()