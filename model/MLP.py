import torch.nn as nn
from torch.optim.lr_scheduler import *
import torch.nn.functional as F
import pickle

class MLP(nn.Module):
    def __init__(self,feature_d):
        super(MLP, self).__init__()
        self.d=feature_d
        self.read_out=nn.Linear(feature_d,1)

    def forward(self,latent_features,target):
        label_features=torch.chunk(latent_features,chunks=latent_features.size(1),dim=1)
        pred=[]
        targ=[]

        pred_t=[]
        for label_feature in label_features:
            pred_t.append(F.sigmoid(self.read_out(label_feature)))
        pred.append(torch.cat(pred_t,1).squeeze(2).float())
        targ.append(target.float())
        return  pred,targ