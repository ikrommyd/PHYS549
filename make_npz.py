import os
import sys
import numpy as np
import uproot
import awkward as ak
from tqdm import tqdm

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
    with uproot.open(f"{file_name}:deepntuplizer/tree;42") as tree:

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
            feature_array, label_array = get_features(f'root_files/{file}')
            final_features = np.vstack((final_features, feature_array))
            final_labels = np.vstack((final_labels, label_array))    
        except FileNotFoundError:
            pass
        
    print(final_features.shape)
    print(final_labels.shape)

    np.savez_compressed(f'root_files/{outfile}.npz', features = final_features, labels = final_labels)

if __name__ == '__main__':

    # take the feature labels out of a root file
    with uproot.open(f"root_files/ntuple_merged_0.root:deepntuplizer/tree;42") as tree:
        features = ['fj_jetNTracks','fj_nSV']+[x for x in tree.keys() if x[:6]=='fj_tau' or x[:8]=='fj_track'] 

    # 2 labels: QCD or Hbb. Logical "and" of labels is used.
    labels = ['fj_isQCD*sample_isQCD',
              'fj_isH*fj_isBB']

    nfeatures = len(features)
    nlabels = len(labels)
    main()