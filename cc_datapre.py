from torch.autograd import Variable
from torch import optim
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import pandas as pd


# The Chemical Checker signature files ["signA1.csv","signA2.csv", ... , "signE5.csv"] were extracted from https://chemicalchecker.org

for i in ["A","B","C","D","E"]:

    df1 = pd.read_csv("./sign"+i+"1.csv",header=0)
    df2 = pd.read_csv("./sign"+i+"2.csv",header=0)
    df3 = pd.read_csv("./sign"+i+"3.csv",header=0)
    df4 = pd.read_csv("./sign"+i+"4.csv",header=0)
    df5 = pd.read_csv("./sign"+i+"5.csv",header=0)

    data_merge =pd.concat([df1,df2.iloc[:,1:],df3.iloc[:,1:],df4.iloc[:,1:],df5.iloc[:,1:]],axis=1)
    data_merge.columns = ['Inchikey']+list(range(640))
    data_merge.to_csv("cc"+i+".csv", index=False, header=True)

class cc_dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.path = data_path
        self.df = pd.read_csv(data_path)

    def __len__(self):
        return len(self.df)
    def __getitem__(self, id):
        compound = self.df.iloc[id][0]
        cc = self.df.iloc[id][1:]
        cc = torch.FloatTensor(cc)
        return compound, cc

class AE(nn.Module):
    def __init__(self):
        super(AE,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64, 24),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 640),
            nn.Tanh()
        )
    def forward(self,x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

batchsize = 128
lr = 3e-4
epoches = 1
weight_decay = 1e-5
datanames = ["ccA","ccB","ccC","ccD","ccE"]
for dataname in datanames:
    model = AE().cuda()
    Data = cc_dataset(dataname+".csv")
    print("dataset")
    dataloader = DataLoader(Data,batch_size=batchsize,shuffle=True)
    optimizier = optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    criterion = nn.MSELoss()
    for epoch in range(epoches):
        model.train()
        for i ,data in enumerate(dataloader):
            x = Variable(data[1].cuda())
            _, decode = model(x)
            loss = criterion(decode,x)
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
            if (i+1)%500 ==0:
                print("epoch:{},loss is {:.3f}".format((epoch+1),loss.data))
    torch.save(model,"./"+dataname+'chemical_checker.pth')

    df = pd.read_csv(dataname+".csv")
    with open("./"+dataname+"_result"+'.csv', 'w') as f:
        f.write('Inchikey\n')
        for i in range(len(df)):
            model.eval()
            cc = torch.FloatTensor(df.iloc[i][1:])
            cc = Variable(cc.cuda())
            encode,_ = model(cc)
            ls = [df.iloc[i][0]]
            ls += encode.tolist()
            f.write(','.join(map(str, ls)) + '\n')

df1 = pd.read_csv("ccA_result.csv",header=0)
df2 = pd.read_csv("ccB_result.csv",header=0)
df3 = pd.read_csv("ccC_result.csv",header=0)
df4 = pd.read_csv("ccD_result.csv",header=0)
df5 = pd.read_csv("ccE_result.csv",header=0)

data_merge = pd.concat([df1, df2.iloc[:, 1:], df3.iloc[:, 1:], df4.iloc[:, 1:], df5.iloc[:, 1:]], axis=1)
data_merge.columns = ['Inchikey'] + list(range(120))
data_merge.to_csv("data/cc_encode.csv", index=False, header=True)