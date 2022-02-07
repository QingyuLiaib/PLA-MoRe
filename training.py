import torch.nn as nn
from torch_geometric.data import DataLoader
from utils import *
from CPA_MoRe import CPA_MoRe

datasets = ["davis",'kiba']
BATCH_SIZE = 48
LR = 0.001
NUM_EPOCHS = 800
acc_step = 16
print("Train the CPA-MoRe model")
print('Benchmarks: {}, Learning rate: {}, Epochs: {}, Batch: {}, Accumulation step: {}'.format(datasets,LR,NUM_EPOCHS,BATCH_SIZE,acc_step))

# define predict function
def predict(model, device, loader):
    model.eval()
    total_preds = torch.Tensor().cuda()
    total_labels = torch.Tensor().cuda()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output,attn = model(data)
            total_preds = torch.cat((total_preds, output), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1)), 0)
    return total_labels.flatten(),total_preds.flatten(),attn


#Training models on the two datasets
for dataset in datasets:

    print('\nTrain the model on '+ dataset + 'dataset')
    train_data = Dataset(root='data', dataset=dataset + '_train')
    valid_data = Dataset(root='data', dataset=dataset + '_validation')
    test_data = Dataset(root='data', dataset=dataset + '_test')

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # training the model
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = CPA_MoRe().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_test_mse = 1000
    best_valid_mse = 1000
    best_epoch = -1
    model_file_name_v = 'model_'   + '_' + dataset + '_valid.model'
    model_file_name_t = 'model_'  + '_' + dataset + '_test.model'
    result_file_name_v = 'result_'  + '_' + dataset + 'v.csv'
    result_file_name_t = 'result_'   + '_' + dataset + 't.csv'
    result_file_name_bt = 'result_'  + '_' + dataset + 'bt.csv'
    for epoch in range(NUM_EPOCHS):
        model.train()
        # gradient accumulation algorithm
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            output, _ = model(data)
            loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
            loss = loss / acc_step
            loss.backward()
            if ((batch_idx + 1) % acc_step) == 0:
                optimizer.step()
                optimizer.zero_grad()
            if batch_idx % 100 == 0:
                print('epoch: [{}/{}] training: {:.1%} loss={:.5f}\n'.format(epoch+1,NUM_EPOCHS,batch_idx / len(train_loader),loss), end='\r')

        print('prediction for validation set')
        val_label, val_pre ,attn= predict(model, device, valid_loader)
        valid_mse = mse(val_label, val_pre).cpu().numpy().tolist()
        print('validation mse: {:.3f}'.format(valid_mse,))
        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file_name_v)
            print('prediction for test set')
            test_label, test_pre, attn = predict(model, device, test_loader)

            test_mse = mse(test_label, test_pre).cpu().numpy().tolist()

            with open(result_file_name_v, 'w') as f:
                f.write('CPA-MoRe reach the best mse: {:.3f} at epoch: {} on validation set'.format(valid_mse,best_epoch))
                f.close()
            with open(result_file_name_t, 'w') as f:
                f.write('The performance of the best model for validation on test set: \n mse: {:.3f}'.format(valid_mse,best_epoch))
                f.close()

        print('dataset: {}'
              'epoch: [{}/{}] validation mse: {:.3f}'
              ' best validation epoch: {} best validation mse {:.3f} test mse: {:.3f} '.format(dataset,
                                                                                                                         epoch + 1,
                                                                                                                         NUM_EPOCHS,
                                                                                                                         valid_mse,
                                                                                                                         best_epoch,
                                                                                                                        best_valid_mse,
                                                                                                                        test_mse))

