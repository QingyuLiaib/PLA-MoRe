# PLA-MoRe: A protein-ligand binding affinity prediction model based on multi-view molecule representations fusion

PLA-MoRe is a deep-learning model to predict protein-ligand binding affinity, which takes the molecular graphs of compounds,the Chemical Checker signatures of compoundsm and the one-dimensional sequences of proteins as input. 

## Data
The Chemical Checker signatures of compounds are extracted from https://chemicalchecker.org


The PDBbind 2016 dataset is extracted from [PDBbind](http://www.pdbbind.org.cn/).

## Usage
data.py: preprocess the data for trainiing, validation and test.


cc_datapre.py: preprocess the CC signatures of compounds.


training.py: train a PLA-MoRe model.

Each benchmark has been divided into three parts for training, validation and test in a proportion of 4:1:1.


For training a PLA-MoRe mode, run: python training.py


