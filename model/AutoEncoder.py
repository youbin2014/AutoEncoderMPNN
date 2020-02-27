import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import *


class AutoEncoder(nn.Module):
    def __init__(self,feature_d):
        super(AutoEncoder, self).__init__()
        self.batch_size=64
        # self.fx0 = nn.Sequential(
        #     nn.Linear(2048, 68 * feature_d),
        #     nn.LeakyReLU(),
        #     nn.Dropout(),
        # )
        self.fx1=nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0),
        )
        self.fx2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0),
        )
        self.fx3 =nn.Sequential(
            nn.Linear(1024, 68*feature_d),
            nn.LeakyReLU(),
            nn.Dropout(0),
        )
        self.fe = nn.LeakyReLU()
        self.d=feature_d
        self.dtype = torch.cuda.FloatTensor # 取消注释以在GPU上运行
        self.Weights=Variable(torch.randn(68,feature_d).type(self.dtype),requires_grad=True)
        self.bias=Variable(torch.randn(68).type(self.dtype),requires_grad=True)
        self.In=nn.InstanceNorm1d(num_features=68)
    def forward(self,features,target):
        # x=self.fx0(features)

        x=self.fx1(features)
        x = self.fx2(x)
        x = self.fx3(x)
        batch_size=x.size(0)
        latent_features1=x.view(-1,68,self.d)
        # target=target.repeat(self.batch_size,1)
        target=target-0.5
        target=target.repeat(1,self.d)
        target=target.view(batch_size,self.d,68)
        target=target.permute(0,2,1).float()
        bias=self.bias.repeat(self.d,1)
        bias = bias.permute(1,0).float()
        Weights=self.Weights.float()
        latent_features2=self.fe(torch.mul(target,Weights)+bias)
        latent_features1=self.In(latent_features1)
        latent_features2=self.In(latent_features2)

        return latent_features1,latent_features2






