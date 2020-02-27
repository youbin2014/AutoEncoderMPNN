from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import numpy as np
from tqdm import tqdm


def val(model,validloader,epoch):
    print('\nEpoch: %d' % epoch)
    model.eval()
    with torch.no_grad():
        with tqdm(total=66) as k:
            for batch_it, (img,label) in enumerate(validloader):
                image = Variable(img.cuda())
                label = Variable(label.cuda())
                pred,targ,feature_x,feature_e = model(image,label)
                # pred,targ= model(image,label)
                ################ calculate output #####################
                if batch_it==0:
                    all_predictions=pred[-1].data.cpu().numpy()
                    all_targets=targ[-1].data.cpu().numpy()
                else:
                    a=pred[-1].data.cpu().numpy()
                    b=targ[-1].data.cpu().numpy()
                    all_predictions=np.concatenate((all_predictions,a),axis=0)
                    all_targets=np.concatenate((all_targets,b),axis=0)
                k.set_description('Step %i' % batch_it)
                k.update(1)

    return all_predictions,all_targets