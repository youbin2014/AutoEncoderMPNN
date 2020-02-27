import torch.nn as nn
from torch.optim.lr_scheduler import *
import torch.nn.functional as F
import pickle

class MPNN(nn.Module):
    def __init__(self,feature_d):
        super(MPNN, self).__init__()
        self.d=feature_d
        self.message_passing=nn.Sequential(nn.Linear(feature_d*2+4,feature_d),nn.LeakyReLU())
        self.update=nn.Sequential(nn.Linear(feature_d,feature_d),nn.LeakyReLU())
        self.read_out=nn.Linear(feature_d,1)
        self.E_trans=nn.Linear(4,feature_d)
        Matrix1= pickle.load(open('/home/hh9665/Desktop/CurrentProject/AutoEncoderMPNN/dataset/Matrix1', 'rb'))
        Matrix2 = pickle.load(open('/home/hh9665/Desktop/CurrentProject/AutoEncoderMPNN/dataset/Matrix2', 'rb'))
        Matrix3 = pickle.load(open('/home/hh9665/Desktop/CurrentProject/AutoEncoderMPNN/dataset/Matrix3', 'rb'))
        Matrix4 = pickle.load(open('/home/hh9665/Desktop/CurrentProject/AutoEncoderMPNN/dataset/Matrix4', 'rb'))
        M1=torch.from_numpy(Matrix1).cuda()
        M2 = torch.from_numpy(Matrix2).cuda()
        M3 = torch.from_numpy(Matrix3).cuda()
        M4 = torch.from_numpy(Matrix4).cuda()
        self.dtype = torch.cuda.FloatTensor # 取消注释以在GPU上运行
        self.E=torch.cat((M1.unsqueeze(0).float(),M2.unsqueeze(0).float(),M3.unsqueeze(0).float(),M4.unsqueeze(0).float()),0)
        self.In=nn.InstanceNorm1d(num_features=1)
        self.T=3
    def forward(self,latent_features,target):
        label_features=torch.chunk(latent_features,chunks=latent_features.size(1),dim=1)
        pred=[]
        targ=[]
        for n in range(self.T):
            new_label_features=[]
            for i in range(68):
                message=label_features[i]
                for j in range(68):
                    if j is not i:
                        EIJ=self.E[:,i,j]
                        message_j=self.message_passing(torch.cat((label_features[j],label_features[i],self.E[:,i,j].repeat(label_features[i].size(0),1,1)),2))
                        message=message+1/68*message_j
                new_label_feature=self.update(message)
                new_label_feature=self.In(new_label_feature)
                new_label_features.append(new_label_feature)
            label_features=new_label_features


        ######read out#######
            pred_t=[]
            for label_feature in label_features:
                pred_t.append(F.sigmoid(self.read_out(label_feature)))
            pred.append(torch.cat(pred_t,1).squeeze(2).float())
            targ.append(target.float())
        return  pred,targ


