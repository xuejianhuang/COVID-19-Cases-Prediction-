
import  torch
import config
from COVID19dataset import  COVID19Dataset
from base_net import  BaseNNet
from torch import  nn
import torch.nn.functional  as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from utils import  plot_learning_curve,plot_pred,save_pred

device=config.device

def train(epoch:int,model:nn.Module,optimizer,train_dataloader):
    model.train()
    total_loss=0
    for idx,(x,target) in enumerate(train_dataloader):
        #print(x)
        x=x.to(device)
        target=target.to(device).to(torch.float32) #    loss.backward()时需要target和pred的数据类型一致，nn.Linear 的输出是float32
        pred=model(x).squeeze(1)
        loss=F.mse_loss(target,pred)
        total_loss+=loss.detach().cpu().item()*len(pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    total_loss = total_loss / len(train_dataloader.dataset)
    print("epoch:{} train loss:{:.6f}".format(epoch,total_loss))
    return total_loss
def eval(model,dev_dataloader):
    model.eval()
    total_loss=0
    preds, targets = [], []
    for idx,(x,target) in enumerate(dev_dataloader):
        x=x.to(device)
        target=target.to(device)
        with torch.no_grad():
            pred = model(x).squeeze(1)
            preds.append(pred)
            targets.append(target)
            loss=F.mse_loss(target,pred)
            total_loss+=loss.detach().cpu().item()*len(pred)
    total_loss=total_loss/len(dev_dataloader.dataset)
    print("eval  loss:{:.6f}".format(total_loss))
    preds = torch.cat(preds, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()
    return total_loss,preds,targets

def test(model,test_dataloader):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x,_ in test_dataloader:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).flatten().numpy()
    return preds


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    model=BaseNNet(93,1)
    optimizer=getattr(torch.optim,config.optimizer)(model.parameters(),**config.optim_hparas)
    n_epoch=config.n_epochs
    loss_record = {'train': [], 'dev': []}  # for recording training loss

    train_dataset = COVID19Dataset(mode="train")
    dev_dataset = COVID19Dataset(mode="dev")
    test_dataset = COVID19Dataset(mode="test")
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    train_loss_list=[]
    dev_loss_list=[]
    min_mse=10000
    early_stop_count=0
    for i in range(n_epoch):
        train_loss=train(i,model,optimizer,train_dataloader)
        dev_loss,_,_=eval(model,dev_dataloader)
        train_loss_list.append(train_loss)
        dev_loss_list.append(dev_loss)
        if dev_loss<min_mse:
            min_mse=dev_loss
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                  .format(i + 1, min_mse))
            torch.save(model.state_dict(),config.save_path)
            early_stop_count=0
        else:
            early_stop_count+=1
        if early_stop_count>=config.early_stop:
            break
    _,preds,targets=eval(model,dev_dataloader)
    plot_learning_curve(train_loss_list, dev_loss_list)
    plot_pred(preds, targets)

    del model
    model = BaseNNet(93,1).to(device)
    model.load_state_dict(torch.load(config.save_path))
    preds=test(model,test_dataloader)
    save_pred(preds,"preds.csv")




