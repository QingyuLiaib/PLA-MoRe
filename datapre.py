import pandas as pd
import json, pickle
import itertools
from collections import OrderedDict
from utils import *
# convert data from GraphDTA and DeepDTA for two benchmarks



# construct a dic for 3_gram protein sequences
seq_dict = {}
mylist = ("".join(x) for x in itertools.product("ACDEFGHIKLMNPQRSTVXWY", repeat=3))
for i in range(9261):
    seq_dict[next(mylist)] = i + 2
compound_iso_smiles = []

dataname = 'data5'

opts = ['training_data5.csv',"valid_data5.csv",'testdata.csv']
for opt in opts:
    df = pd.read_csv(opt)
    compound_iso_smiles += list( df['SMILES'] )
compound_iso_smiles = set(compound_iso_smiles)

smile_graph = {}
for smile in compound_iso_smiles:
    smile_graph[smile] = smile_to_graph(smile)



df = pd.read_csv('training_'+dataname+'.csv')
train_drugs, train_prots, train_inchikey, train_Y ,train_prots_name= list(df['SMILES']),list(df['proseq']),list(df['INCHIKEY']),list(df['affinity']),list(df['pdbname'])
XT = [seq2ngram(t,3,seq_dict=seq_dict) for t in train_prots]
train_drugs, train_prots, train_inchikey ,train_Y = np.asarray(train_drugs), np.asarray(XT),np.asarray(train_inchikey), np.asarray(train_Y)

df = pd.read_csv('valid_'+dataname+'.csv')
validation_drugs, validation_prots,validation_inchikey,validation_Y ,validation_prots_name= list(df['SMILES']),list(df['proseq']),list(df['INCHIKEY']),list(df['affinity']),list(df['pdbname'])
XT = [seq2ngram(t,3,seq_dict=seq_dict) for t in validation_prots]
validation_drugs, validation_prots, validation_inchikey, validation_Y = np.asarray(validation_drugs), np.asarray(XT),np.array(validation_inchikey), np.asarray(validation_Y)

df = pd.read_csv('testdata.csv')
test_drugs, test_prots,test_inchikey,   test_Y ,test_prots_name= list(df['SMILES']),list(df['proseq']),list(df['INCHIKEY']),list(df['affinity']),list(df['pdbname'])
XT = [seq2ngram(t,3,seq_dict=seq_dict) for t in test_prots]
test_drugs, test_prots, test_inchikey, test_Y = np.asarray(test_drugs), np.asarray(XT),np.array(test_inchikey), np.asarray(test_Y)

CCdic = json.loads(open("cc_dic.txt", 'r').read())

train_data = Dataset(root='data', dataset=dataname+'_train', xd=train_drugs, xt=train_prots, dic = CCdic,inchikey = train_inchikey,y=train_Y,smile_graph=smile_graph)
validation_data = Dataset(root='data', dataset=dataname+'_validation', xd=validation_drugs, xt=validation_prots, dic=CCdic,inchikey = validation_inchikey,y=validation_Y,smile_graph=smile_graph)
test_data = Dataset(root='data', dataset='data_test', xd=test_drugs, xt=test_prots, dic=CCdic,inchikey = test_inchikey,y=test_Y,smile_graph=smile_graph)
