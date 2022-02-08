# CPA-MoRe: A compound-protein binding affinity prediction model based on multi-view molecule representations fusion

CPA-MoRe is a deep-learning model to predict compound-protein binding affinity, which takes the molecular graphs of compounds,the Chemical Checker signatures of compoundsm and the one-dimensional sequences of proteins as input. 

## Data
The Chemical Checker signatures of compounds are extracted from https://chemicalchecker.org


The Davis and KIBA datasets are extracted from DeepDTA.

## Usage
data.py: preprocess the data for trainiing, validation and test.


cc_datapre.py: preprocess the CC signatures of compounds.


training.py: train a CPA-MoRe model.

Each benchmark has been divided into three parts for training, validation and test in a proportion of 4:1:1.


For training a CPA-MoRe mode, run: python training.py


