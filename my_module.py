# my_module.py
import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
def visualize2DSoftMax(X:np.ndarray,y:np.ndarray,model,title=None,nPoints:int=20):
    x_min=np.min(X[:,0])-0.5
    x_max=np.max(X[:,0])+0.5
    y_min=np.min(X[:,1])-0.5
    y_max=np.max(X[:,1])+0.5
    xv,yv = np.meshgrid(np.linspace(x_min,x_max,nPoints),np.linspace(y_min,y_max,nPoints))
    xy=np.hstack((xv.reshape(-1,1),yv.reshape(-1,1)))
    with torch.no_grad():
        logits=model(torch.tensor(xy,dtype=torch.float32))
        y_hat=torch.nn.functional.softmax(logits,dim=1).numpy()
    plt.contourf(xv,yv,y_hat[:,0].reshape(nPoints,nPoints),levels=np.linspace(0,1,30),cmap=plt.cm.RdYlBu)
    sns.scatterplot(x=X[:,0],y=X[:,1],hue=y,style=y,ax=plt.gca())
    plt.gca().set_title(title)
    
def train_simple(model:torch.nn.Module,loss_func,optimizer:torch.optim.Optimizer,dataloader:DataLoader,nEpochs:int=20,results=None):
    for epoch in tqdm(range(nEpochs),desc="Train Batch"):
        model.train()
        running_loss=0
        for inputs,labels in tqdm(dataloader,desc="Batch",leave=False):
            optimizer.zero_grad()
            y_hat=model(inputs)
            loss=loss_func(y_hat,labels)
            loss.backward()
            optimizer.step()
            

def save_snapshot(model:torch.nn.Module,optimizer:torch.optim.Optimizer,results,file):
    torch.save({
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'results': results
    },file)
