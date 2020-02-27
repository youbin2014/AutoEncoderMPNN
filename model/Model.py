import torch.nn as nn
from torch.optim.lr_scheduler import *
from model.resnet import resnet50
import torch.nn.functional as F


resnet=resnet50(pretrained=True)

class Model(nn.Module):
    def __init__(self,resnet,AutoEncoder,MPNN,TRAIN=True,feature_d=16):
        super(Model, self).__init__()
        self.TRAIN=TRAIN
        self.d=feature_d
        self.resnet_layer = nn.Sequential(*list(resnet.children())[:-1])
        self.linear=nn.Linear(2048,68)
        self.MPNN=MPNN(self.d)
        # self.MLP=MLP(self.d)
        self.AutoEncoder=AutoEncoder(self.d)

    def forward(self, image, target):
        if not self.TRAIN:
            target=torch.zeros(68,self.d)
        # pred=[]
        # targ=[]
        x=self.resnet_layer(image)
        x=x.view(x.size(0), -1)
        # x=self.linear(x)
        # x=F.sigmoid(x)
        Fx,Fe=self.AutoEncoder(x,target.clone())
        pred,targ=self.MPNN(Fx,target)
        # pred.append(x.float())
        # targ.append(target.float())
        # return pred,targ,Fx,Fe
        return pred,targ,Fx,Fe










