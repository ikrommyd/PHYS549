# PHYS 549 group project: <br> Hbb jet tagging with CMS open data


This repository is a shared workspace for the PHYS 549 group project.
The goal of this project is to classify jets as originating from Higgs to bb decay or as originating from QCD multijet production.\
The dataset is from http://opendata.cern.ch/record/12102 ([DOI:10.7483/OPENDATA.CMS.JGJX.MS7Q](http://doi.org/10.7483/OPENDATA.CMS.JGJX.MS7Q)).

A very basic model is so far implemented in the `train.ipynb` notebook.\
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
