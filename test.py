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

from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

#################################### Preprocessing #################################################
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

testset=Privacy_Data(transform=transform_test,train=False, test=True, valid=False)
testloader=torch.utils.data.DataLoader(testset,batch_size=64,shuffle=False,num_workers=8)


resnet = resnet50(pretrained=True)
model=Model(resnet,AutoEncoder,MPNN,TRAIN=True,feature_d=16)
model.cuda()
model.load_state_dict(torch.load('result/' + 'T20200221_0/'+ '/model0.pth'))
def test(model,trainloader):
    model.eval()
    with torch.no_grad():
        with tqdm(total=66) as t:
            for batch_it, (img,label) in enumerate(trainloader):
                image = Variable(img.cuda())
                label = Variable(label.cuda())
                pred,targ,feature_x,feature_e = model(image,label)

                ################ calculate output #####################
                if batch_it==0:
                    all_predictions=pred[-1].data.cpu().numpy()
                    all_targets=targ[-1].data.cpu().numpy()
                else:
                    a=pred[-1].data.cpu().numpy()
                    b=pred[-1].data.cpu().numpy()
                    all_predictions=np.concatenate((all_predictions,a),axis=0)
                    all_targets=np.concatenate((all_targets,b),axis=0)

    return all_predictions,all_targets

if __name__ == '__main__':
    all_predictions,all_targets=test(model,trainloader=testloader)
    test_metrics, label_sum, allAP = evals.compute_metrics(all_predictions, all_targets, 1, 1, 1, all_metrics=True)

