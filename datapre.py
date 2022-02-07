import pandas as pd
import json, pickle
import itertools
from collections import OrderedDict
from utils import *
# convert data from GraphDTA and DeepDTA for two benchmarks

datasets = ["kiba",'davis']
for dataset in datasets:
    fpath = 'data/' + dataset + '/'
    train_fold = json.load(open(fpath + "folds/"+dataset+"_train_setting1.txt"))
    valid_fold = json.load(open(fpath + "folds/"+dataset+"_validation_setting1.txt"))
    test_fold = json.load(open(fpath + "folds/"+dataset+"_test_setting.txt"))
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')
    drugs = []
    prots = []
    prots_name = []
    INCHIKEY = []
    #generate inchikey for each ligand
    for d in ligands.keys():
        mol = Chem.MolFromSmiles(ligands[d])
        lg = Chem.MolToSmiles(mol, isomericSmiles=False)
        drugs.append(lg)
        key = Chem.inchi.MolToInchiKey(mol)
        INCHIKEY.append(key)
    for t in proteins.keys():
        prots.append(proteins[t])
        prots_name.append(t)
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)
    cc_dic = {}

    # generate features for ligends from the chemical checker
    df = pd.read_csv("data/cc_encode.csv",header=0)
    for i in INCHIKEY:
        loc = np.where(df["Inchikey"] == i)
        if len(list(loc)[0]) ==0:
            cc_dic[i] = str(np.array([]).tolist())
            continue
        temp = np.array(df.iloc[loc])
        cc_dic[temp[0, 0]] = temp[0, 1:].tolist()
    json_str = json.dumps(cc_dic)
    with open("data/"+dataset+"/cc_dic.txt", 'w',encoding='utf-8') as file:
        file.write(json_str)

    opts = ['train', 'test',"validation"]
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity) == False)
        if opt == 'train':
            rows, cols = rows[train_fold], cols[train_fold]
        elif opt == 'validation':
            rows, cols = rows[valid_fold], cols[valid_fold]
        elif opt == 'test':
            rows, cols = rows[test_fold], cols[test_fold]
        with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,INCHIKEY,prots_name,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [INCHIKEY[rows[pair_ind]]]
                ls += [prots_name[cols[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                f.write(','.join(map(str, ls)) + '\n')

# construct a dic for 3_gram protein sequences
seq_dict = {}
mylist = ("".join(x) for x in itertools.product("ACDEFGHIKLMNPQRSTVXWY", repeat=3))
for i in range(9261):
    seq_dict[next(mylist)] = i + 2
compound_iso_smiles = []

for dt_name in ['kiba','davis']:
    opts = ['train','test',"validation"]
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        compound_iso_smiles += list( df['compound_iso_smiles'] )
compound_iso_smiles = set(compound_iso_smiles)

smile_graph = {}
for smile in compound_iso_smiles:
    smile_graph[smile] = smile_to_graph(smile)

# convert to PyTorch data format
datasets = ['davis','kiba']
for dataset in datasets:

    df = pd.read_csv('data/' + dataset + '_train.csv')
    train_drugs, train_prots, train_inchikey, train_Y ,train_prots_name= list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['INCHIKEY']),list(df['affinity']),list(df['prots_name'])
    XT = [seq2ngram(t,3,seq_dict=seq_dict) for t in train_prots]
    train_drugs, train_prots, train_inchikey ,train_Y = np.asarray(train_drugs), np.asarray(XT),np.asarray(train_inchikey), np.asarray(train_Y)

    df = pd.read_csv('data/' + dataset + '_validation.csv')
    validation_drugs, validation_prots,validation_inchikey,validation_Y ,validation_prots_name= list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['INCHIKEY']),list(df['affinity']),list(df['prots_name'])
    XT = [seq2ngram(t,3,seq_dict=seq_dict) for t in validation_prots]
    validation_drugs, validation_prots, validation_inchikey, validation_Y = np.asarray(validation_drugs), np.asarray(XT),np.array(validation_inchikey), np.asarray(validation_Y)

    df = pd.read_csv('data/' + dataset + '_test.csv')
    test_drugs, test_prots,test_inchikey,   test_Y ,test_prots_name= list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['INCHIKEY']),list(df['affinity']),list(df['prots_name'])
    XT = [seq2ngram(t,3,seq_dict=seq_dict) for t in test_prots]
    test_drugs, test_prots, test_inchikey, test_Y = np.asarray(test_drugs), np.asarray(XT),np.array(test_inchikey), np.asarray(test_Y)

    CCdic = json.loads(open("data/" + dataset + "/cc_dic.txt", 'r').read())

    print('preparing ', dataset + ' in pytorch format!')
    train_data = Dataset(root='data', dataset=dataset+'_train', xd=train_drugs, xt=train_prots, dic = CCdic,inchikey = train_inchikey,y=train_Y,smile_graph=smile_graph)
    validation_data = Dataset(root='data', dataset=dataset+'_validation', xd=validation_drugs, xt=validation_prots, dic=CCdic,inchikey = validation_inchikey,y=validation_Y,smile_graph=smile_graph)
    test_data = Dataset(root='data', dataset=dataset+'_test', xd=test_drugs, xt=test_prots, dic=CCdic,inchikey = test_inchikey,y=test_Y,smile_graph=smile_graph)

