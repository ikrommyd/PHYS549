'''
This script extracts desired features out of a range of .root files from the "root_files" folder and saves them into a numpy npz file.
The desired features must be defined in a list called "features" after the imports block.
By default we have selected 49 features, the number of jets and tracks in an event, 18 kinematic variables, and 29 shape variables.

Usage: python make_npz.py <starting root file number> <ending root file number> <desired npz file name>
Example: python make_npz.py 0 8 myarrays
This will the the extract the desired features from the root files "ntuple_merged_0.root" to "ntuple_merged_8.root" in order and save them into myarrays.npz
'''

import os
import sys
import numpy as np
import uproot
import awkward as ak
from tqdm import tqdm

# define the features that we want to extract
features1 = ["fj_jetNTracks",
            "fj_nSV",
            "fj_eta",
            "fj_mass",
            "fj_phi",
            "fj_pt",
            "fj_ptDR",
            "fj_relptdiff",
            "fj_sdn2",
            "fj_sdsj1_eta",
            "fj_sdsj1_mass",
            "fj_sdsj1_phi",    
            "fj_sdsj1_pt",
            "fj_sdsj1_ptD",
            "fj_sdsj2_eta",
            "fj_sdsj2_mass",
            "fj_sdsj2_phi",
            "fj_sdsj2_pt",
            "fj_sdsj2_ptD",
            "fj_z_ratio"]

with uproot.open(f"../root_files/ntuple_merged_0.root:deepntuplizer/tree") as tree:
    features2 = [x for x in tree.keys() if x[:6]=='fj_tau' or x[:8]=='fj_track']

features = features1 + features2


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

def get_features(file_name):
    '''
    Function that extracts our chosen feature and label arrays from a root file
    for the events that are labeled as QCD or Hbb and returns two 2D arrays.
    The first array is the features array and has the shape (nummber_of_events, number_of_features).
    The second array is the labels array ans has the shape (number_of_events, 2)
    '''
    with uproot.open(f"{file_name}:deepntuplizer/tree") as tree:

        feature_array = np.stack(list(tree.arrays(features,library='np').values()),axis=-1)
        label_array = np.stack([get_labels(tree,i) for i in labels],axis=-1)
        feature_array = feature_array[np.sum(label_array,axis=1)==1]
        label_array = label_array[np.sum(label_array,axis=1)==1]

    return feature_array, label_array


def main():

    start = int(sys.argv[1])
    end = int(sys.argv[2]) + 1
    outfile = sys.argv[3]


    final_features = np.empty((0,nfeatures))
    final_labels = np.empty((0,nlabels))

    for i in tqdm(range(start, end)):
        file = f"ntuple_merged_{i}.root"
        try:
            feature_array, label_array = get_features(f'../root_files/{file}')
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