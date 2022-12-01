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

with uproot.open(f"../root_files/ntuple_merged_0.root:deepntuplizer/tree") as tree:
    features = [x for x in tree.keys() if x[:5]=='track']



def reshape(awk_ar,sorter,fet_lngth, ascending=False):
    '''
    This function creates takes an awkward array of track information in an event, and formats it in a standard 2D array.
    The tracks of an event are sorted according to the 'sorter' argument in descending order (greatest value first).
    Events that have too many tracks are truncated, while those with too few are padded with zeros afterwards.
    The returned array is an array of 2D arrays containing  track information for each event.
    '''
    evt_ar = [[] for i in range(len(awk_ar[0]))]
    srt_idx = np.where(np.array(features)==sorter)[0][0]
    
    for fet in awk_ar:
        for evt_n in range(len(fet)):
            evt_ar[evt_n].append(np.array(fet[evt_n]))
            
    # Making 2D arrays for each event, where each each row of that event's 2D array is now a feature array
    evt_ar2d = []
    
    for evt in evt_ar:
        evt_ar2d.append(np.stack(evt))
        
    # Now sorting each 2d array's columns according to the 'sorter' feature
    evt_ar2d_srtd = []
    
    for evt in evt_ar2d:
        idcs = evt[srt_idx].argsort()[::-1]
        evt_ar2d_srtd.append(evt[:,idcs])
        
    # Now need to standardize feature array per event
    evt_std = []
    npad = 0
    
    for evt in evt_ar2d_srtd:
        if len(evt[0]) > fet_lngth:
            evt_std.append( np.swapaxes(evt[:,:fet_lngth],0,1) )
#             evt_std.append( evt[:,:fet_lngth].flatten() )
        else:
            npad += 1
            padded = np.pad(evt,((0,0),(0,fet_lngth - len(evt[0]))))
            evt_std.append( np.swapaxes(padded,0,1) )
#             evt_std.append( np.pad(evt,((0,0),(0,fet_lngth - len(evt[0])))).flatten() )
            
    print('Number of events padded: {}'.format(npad))
    
    return evt_std

def get_features(file_name, var_sort, ntrx):
    '''
    Function that extracts our chosen feature and label arrays from a root file
    for the events that are labeled as QCD or Hbb and returns two 2D arrays.
    The first array is the features array and has the shape (nummber_of_events, number_of_features).
    The second array is the labels array ans has the shape (number_of_events, 2)
    '''
    with uproot.open(f"{file_name}:deepntuplizer/tree") as tree:

        fet_data = []
        lbl_data = []
        
        for fet in features:
            fet_data.append(np.array(tree[fet]))
            
        for lbl in labels:
            lbl_data.append(np.array(tree[lbl]))

        feature_array = reshape(fet_data, var_sort, ntrx)

        # This part organizes it by event, rather than feature. (features move from rows to columns)
        label_array = np.stack(lbl_data,axis=-1)
        
        #feature_array = np.array(feature_array)[label_array.any(1)]
        feature_array = np.array(feature_array)
        label_array = label_array
        
    #     return label_array
    return feature_array, label_array


def main():

    start = int(sys.argv[1])
    end = int(sys.argv[2]) + 1
    outfile = sys.argv[3]
    srtr = sys.argv[4]
    ntrx = in(sys.argv[5]


    final_features = np.empty((0,nfeatures))
    final_labels = np.empty((0,nlabels))

    for i in tqdm(range(start, end)):
        file = f"ntuple_merged_{i}.root"
        try:
            feature_array, label_array = get_features(f'../root_files/{file}',srtr,ntrx)
            #final_features = np.vstack((final_features, feature_array))
            #final_labels = np.vstack((final_labels, label_array))    
        except FileNotFoundError:
            pass
        
    print(features_array.shape)
    print(label_array.shape)

    np.savez(f'../root_files/{outfile}.npz', features = features_array, labels = label_array, names = features)

if __name__ == '__main__':

    # 2 labels: QCD or Hbb. Logical "and" of labels is used.
    labels = ['sample_isQCD',
              'label_H_bb']

    nfeatures = len(features)
    nlabels = len(labels)
    main()
