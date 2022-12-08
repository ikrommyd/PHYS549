# PHYS 549 group project: <br> Hbb jet tagging with CMS open data


This repository is a shared workspace for the PHYS 549 group project.
The goal of this project is to classify jets as originating from Higgs to bb decay or as originating from QCD multijet production.\
The dataset is from http://opendata.cern.ch/record/12102 ([DOI:10.7483/OPENDATA.CMS.JGJX.MS7Q](http://doi.org/10.7483/OPENDATA.CMS.JGJX.MS7Q)).

To run this repository you need to create a conda environment with the necessary dependencies provided by the `environment.yml` file.\
After cloning the repository, just create an environment by running 
```
conda env create -f environment.yml
```
If you don't have conda, you can get it at https://docs.conda.io/en/latest/miniconda.html. \
You can then activate the environment by running 
```
conda activate phys549
```
To run the repository you also need tha data `.root` files inside a folder called `root_files`.\
We provide index `.txt` files with links to download the data.\
WARNING: The files are roughly 100 GB so make sure you have enough space and time.\
To download all of the `.root` files with `wget`, just do
```
cd root_files
wget -i HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC_test_root_file_index_wget.txt
wget -i HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC_train_root_file_index_wget.txt
```
or you can also use XRootD if you have it installed
```
cd root_files
xrdcp -I HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC_test_root_file_index_xrdcp.txt .
xrdcp -I HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC_train_root_file_index_xrdcp.txt .
```

## Exploration and Modeling

We have implemented the data exploration and modeling in the `explore.ipynb` and `train_models.ipynb` notebooks respectively.\
The folder `scripts` contains some `.py` files that define a neural network model class, a plotting function and two data preprocessing scripts.
These `.py` files are used by the notebooks for the modeling and for preprocessing before the notebooks are even able to run.

To use these notebooks, some preprocessing must take place first.
The script `make_npz.py` extracts desired features out of a range `.root` files from the `root_files` folder and saves them into a numpy `.npz` file.
The feature names are available at http://opendata.cern.ch/record/12102.
The desired features to extract must be defined in a list called `features` after the imports block in the script.
By default we have selected 49 features, the number of jets and tracks in an event, 18 kinematic variables, and 29 shape variables.

Usage of this script:
```
cd scripts
python make_npz.py <starting root file number> <ending root file number> <desired npz file name>
```
Example:
```
cd scripts
python make_npz.py 0 8 myarrays
```
This will the the extract the desired features from the `.root` files from `ntuple_merged_0.root` to `ntuple_merged_8.root` in order and save them into `myarrays.npz` within the `root_files` folder.

In our modeling we used all of the `.root` files and split them into 3 sets. After downloading all the files, this was done by:
```
cd scripts
python make_npz.py 0 19 combined_validate
python make_npz.py 20 80 combined_train
python make_npz.py 81 90 combined_test
```
which will create 3 `.npz` files for training, validation and testing purposes using all 91 `.root` files.

Now the notebook `explore.ipynb` can be used which will create a file called `train_test_49variables.npz` which will contain the final training and testing arrays. This notebook is open to modification to the user's preferences but can also be run as is.

Finally, after this final `.npz` file is created, the notebook `train_models.ipynb` can be run to train and validate some models.
We have implemented a Neural Network classifier, a Gradieng Boosting Trees classifier, and a Linear Discriminant Analysis classifier and all perform very well.

The output plots after your runs can all be found within the `plots` folder which already contains some of our results.

After the training and validation is done, the best parameters of the models are saved within the `model_checkpoints` directory.
The models can now be tested on new data that we have never looked at before.
In order to do this, you must first run the `preprocess_testing.ipynb` notebook to create a file called `test_49variables.npz` which is similar to the `train_test_49variables.npz` file, but contains new unexplored data.
Then the `test_models.ipynb` notebook can be run which will test the pretrained models unto the new data.

## Using features of individual tracks of jets

We also perform modeling using variables of individual tracks within the jet events.
This is implemented in a similar manner in the `train_models_tracks.ipynb` and `test_models_tracks.ipynb` notebooks.

In order to run these notebooks you must first use the script `make_npz_tracks.py` in a similar manner to the `make_npz.py` script.
This script will extract 28 variables from the 10 tracks of each jet event with the highest momentum.
If there are less than 10 tracks, zeros will be filled.
````
cd scripts
python make_npz_tracks.py 0 19 combined_tracks_validate
python make_npz_tracks.py 20 80 combined_tracks_train
python make_npz_tracks.py 81 90 combined_tracks_test
````
After this preprocessing, the notebooks `train_models_tracks.ipynb` and `test_models_tracks.ipynb` can be run in order.
These will train a multilayer perceptron using those track variables and then test it on unexplored data.