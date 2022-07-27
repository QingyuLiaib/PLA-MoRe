import os
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import torch
from rdkit import Chem
import networkx as nx
from math import sqrt
from scipy import stats


def atom_features(atom):
   #Generate features for a atom, which are 45-dim vectors
    atoms = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Si', 'I', 'B', 'Se', 'Unknown']
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2,
                              'other']
    atom_feature =  one_of_k_encoding_unk(atom.GetSymbol(),atoms) +\
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) +\
                    one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType)+\
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +\
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +\
                    [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] +\
                    [atom.GetIsAromatic()]
    return np.array(atom_feature)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    compound_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature)
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return compound_size, features, edge_index


def seq2ngram(sequence, ngram,seq_dict,max_len = 1000):
    words = [sequence[i:i+ngram] for i in range(len(sequence)-ngram+1)]
    x = np.ones(max_len)
    x[0]= 0
    for i, ch in enumerate(words[:max_len-1]):
        x[i+1] = seq_dict[ch]
    return x

def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse

def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

class Dataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xt=None,dic=None,inchikey=None,y=None, transform=None,
                 pre_transform=None,smile_graph=None):
        super(Dataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, dic,inchikey,y,smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    def process(self, xd, xt, dic,inchikey,y,smile_graph):
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            smiles = xd[i]
            cc = dic[inchikey[i]]
            target = xt[i]
            labels = y[i]
            c_size, features, edge_index = smile_graph[smiles]
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]),
                                cc=torch.FloatTensor([cc]),
                                target = torch.LongTensor([target]))
            data_list.append(GCNData)
            print(GCNData)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs