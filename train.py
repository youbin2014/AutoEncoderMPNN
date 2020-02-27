from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


def train(model,scheduler,optimizer,trainloader,epoch,alpha,gamma):
    print('\nEpoch: %d' % epoch)
    scheduler.step()
    model.train()
    with tqdm(total=157) as t:
        for batch_it, (img,label) in enumerate(trainloader):
            image = Variable(img.cuda())
            label = Variable(label.cuda())
            optimizer.zero_grad()
            pred,targ,feature_x,feature_e = model(image,label)
            loss,edloss=0,0
            for i in range(len(pred)):
                loss+=F.binary_cross_entropy(pred[i],targ[i])
            edloss=F.mse_loss(feature_x,feature_e)
            # if epoch<=1:
            #     Loss=edloss
            # else:
            Loss=loss+alpha*edloss
            Loss.backward()
            optimizer.step()

            ########## Matrix loss ###########

            t.set_description('Step %i' % batch_it)
            t.set_postfix(Loss=Loss.data.cpu().numpy(),BCE_loss=loss.data.cpu().numpy(),AutoEncoder_loss=edloss.data.cpu().numpy())
            t.update(1)

            ################ calculate output #####################
            if batch_it==0:
                all_predictions=pred[-1].data.cpu().numpy()
                all_targets=targ[-1].data.cpu().numpy()
            else:
                a=pred[-1].data.cpu().numpy()
                b=targ[-1].data.cpu().numpy()
                all_predictions=np.concatenate((all_predictions,a),axis=0)
                all_targets=np.concatenate((all_targets,b),axis=0)
    losses={'total_loss':Loss.mean().data.cpu().numpy(),
        'BCEloss':loss.mean().data.cpu().numpy(),
        'AutoEncoder_loss': edloss.mean().data.cpu().numpy(),
        }
    return all_predictions,all_targets,losses