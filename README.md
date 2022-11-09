# PHYS 549 group project: <br> Hbb jet tagging with CMS open data


This repository is a shared workspace for the PHYS 549 group project.
The goal of this project is to classify jets as originating from Higgs to bb decay or as originating from QCD multijet production.\
The dataset is from http://opendata.cern.ch/record/12102 ([DOI:10.7483/OPENDATA.CMS.JGJX.MS7Q](http://doi.org/10.7483/OPENDATA.CMS.JGJX.MS7Q)).

A very basic example model is implemented in the `train_basic.ipynb` notebook.\
This model is a very simple multilayer perceptron that trains on one `.root` file and tests on another.

To run this notebook you need to create a conda environment with the necessary dependencies provided by the `environment.yml` file.\
After cloning the repository, just create an environment by running 
```
conda env create -f environment.yml
```
If you don't have conda, you can get it at https://docs.conda.io/en/latest/miniconda.html. \
You can then activate the environment by running 
```
conda activate phys549
```
To run the notebook you also need tha data `.root` files inside a folder called `root_files`.\
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
You can also download individual `.root` files instead of all of them by using specific links within the `.txt` files.\
For instance:
```
cd root_files
wget http://opendata.cern.ch/eos/opendata/cms/datascience/HiggsToBBNtupleProducerTool/HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/test/ntuple_merged_0.root
wget http://opendata.cern.ch/eos/opendata/cms/datascience/HiggsToBBNtupleProducerTool/HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/test/ntuple_merged_10.root
```
or
```
cd root_files
xrdcp root://eospublic.cern.ch//eos/opendata/cms/datascience/HiggsToBBNtupleProducerTool/HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/test/ntuple_merged_0.root .
xrdcp root://eospublic.cern.ch//eos/opendata/cms/datascience/HiggsToBBNtupleProducerTool/HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/test/ntuple_merged_10.root .
```
These are the two files that the notebook uses by default but you can use any other set of files or multiple files if you modify the code.

## Exploration and Modeling

We have implemented some data exploration and modeling so far in the `explore.ipynb` and `model.ipynb` notebooks respectively.\
We note that the modeling will later be ported into `.py` scripts when it is finalized but we keep using notebooks for now since a lot of things change often and we want to be able to visualize things quickly.

To use these notebooks, some preprocessing must take place first.
The script `make_npz.py` extracts desired features out of a range `.root` files from the `root_files` folder and saves them into a numpy `.npz` file.
The feature names are available at http://opendata.cern.ch/record/12102.
The desired features to extract must be defined in a list called `features` after the imports block in the script.
By default we have selected 49 features, the number of jets and tracks in an event, 18 kinematic variables, and 29 shape variables.

Usage of this script:
```
python make_npz.py <starting root file number> <ending root file number> <desired npz file name>
```
Example:
```
python make_npz.py 0 8 myarrays
```
This will the the extract the desired features from the `.root` files from `ntuple_merged_0.root` to `ntuple_merged_8.root` in order and save them into `myarrays.npz` within the `root_files` folder.

In our modeling we used all of the `.root` files and split them into 3 sets. After downloading all the files, this was done by:
```
python make_npz.py 0 8 combined_test
python make_npz.py 9 80 combined_train
python make_npz.py 81 90 combined_validate
```
which will create 3 `.npz` files for training, testing and validation purposes using all 91 `.root` files.

Now the notebook `explore.ipynb` can be used which will create a file called `train_test_49variables.npz` which will contain the final training and testing arrays. This notebook is open to modification to the user's preferences but can also be run as is.

Finally, after this final `.npz` file is created, the notebook `model.ipynb` can be run to train and test some models.
So far we have implemented a Neural Network classifier, a Gradieng Boosting Trees classifier, and a Linear Discriminant Analysis classifier and all perform very well.

The output plots after your runs can all be found with the `plots` folder which already contains some of our results.