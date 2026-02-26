import numpy as np
import torch as torch

def visualize2DSoftMax(X:np.ndarray,y:np.ndarray,model,title=None,nPoints:long=20):
    x_min=np.min(X[:,0])-0.5
    x_max=np.max(X[:,0])+0.5
    y_min=np.min(X[:,1])-0.5
    y_max=np.max(X[:,1])+0.5
    xv,yv = np.meshgrid(np.linspace(x_min,x_max,nPoints),np.linspace(y_min,y_max,nPoints))
    xy=np.hstack((xv.reshape(-1,1),yv.reshape(-1,1)))
    with torch.no_grad():
        logits=model(torch.tensor(xy,dtype=torch.float32))
        y_hat=torch.nn.functional.softmax(logits,dim=1).numpy()
    
    cs