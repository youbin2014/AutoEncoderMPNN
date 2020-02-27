import torch.optim as optim
from torch.optim.lr_scheduler import *
import torchvision.transforms as transforms
from dataset.Privacy_Data import Privacy_Data
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from model.resnet import resnet50
from tools import evals
from model.Model import Model
from model.AutoEncoder import AutoEncoder
from model.MPNN import MPNN
from model.MLP import MLP
from train import train
from valid import val
from tools.SaveandSee import SaveandSee
from tools.SendEmail import send_email
import torch.nn as nn


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

#################################### Preprocessing #################################################
transform_train=transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.RandomCrop((224,224)),
    # transforms.RandomResizedCrop(224),
    # transforms.Resize((224,224)),
    transforms.RandomRotation((-5,5)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

transform_val=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

trainset = Privacy_Data(transform=transform_train, train=True, test=False, valid=False)
valset = Privacy_Data(transform=transform_val, train=False, test=False, valid=True)
trainloader = DataLoaderX(trainset, batch_size=64, shuffle=True, num_workers=8)
valloader = DataLoaderX(valset, batch_size=64, shuffle=False, num_workers=8)


resnet = resnet50(pretrained=True)
resnet.fc=nn.Linear(2048,68)
resnet.load_state_dict(torch.load('/home/hh9665/Desktop/CurrentProject/AutoEncoderMPNN/ckp/model_resnet50_my_loss1.pth'))
model=Model(resnet,AutoEncoder,MPNN,TRAIN=True,feature_d=16)
print(model)
    # model.load_state_dict(torch.load('/home/hh9665/Desktop/CurrentProject/AutoEncoderMPNN/result/T20200224_0/model0.pth'))
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=0)
scheduler = StepLR(optimizer, step_size=3, gamma=0.9)
N_epoch=80

RESULTS = {'LOSS': [], 'BCELOSS': [], 'AUTOENCODER_LOSS': [],
           'ACC': [], 'mAP': [], 'miF1': [], 'maF1': [], 'LOSS_val': [],
           'ACC_val': [], 'mAP_val': [], 'miF1_val': [], 'maF1_val': [],
           'Epoch': [], 'results_dir': '/home/hh9665/Desktop/CurrentProject/AutoEncoderMPNN/result' + '/T20200225_0' + '/'}
send_email(train,RESULTS)
for epoch in range(N_epoch):
    ############# Train ################
    # if epoch>1:
    for name, p in model.named_parameters():
        if name.startswith('resnet'):
            p.requires_grad = False

    optimizer = optim.Adam(filter(lambda x: x.requires_grad is not False, model.parameters()), lr=0.00005)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.9)
    # if epoch>=5:
    #     for name, p in model.named_parameters():
    #         if name.startswith('AutoEncoder'):
    #             p.requires_grad = True
    #     optimizer = optim.Adam(filter(lambda x: x.requires_grad is not False, model.parameters()), lr=0.00005)
    #     scheduler = StepLR(optimizer, step_size=3, gamma=0.9)

    all_predictions, all_targets, losses = train(model, scheduler, optimizer, trainloader, epoch, alpha=1,gamma=0)
    train_metrics, label_sum, allAP = evals.compute_metrics(all_predictions, all_targets, 1, 1, 1, all_metrics=True)
    RESULTS = SaveandSee(RESULTS, train_metrics, allAP, losses=losses, type='train', epoch=epoch)
    send_email(train, RESULTS)
    ############# Val ################
    all_predictions, all_targets = val(model, valloader, epoch, )
    val_metrics, label_sum, allAP = evals.compute_metrics(all_predictions, all_targets, 1, 1, 1, all_metrics=True)
    RESULTS = SaveandSee(RESULTS, val_metrics, allAP, losses=losses, type='val', epoch=epoch)
    send_email(val, RESULTS)
    torch.save(model.state_dict(), 'result/' + 'T20200225_0/'+ '/model0.pth')

