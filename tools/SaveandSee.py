import numpy as np
import os
import visdom
import pickle

vis=visdom.Visdom()

def SaveandSee(RESULTS,metrics,allAP,losses={},type='train',epoch=0):
	if not os.path.exists(RESULTS['results_dir']):
		os.makedirs(RESULTS['results_dir'])
	if type=='train':
		RESULTS['LOSS'].append(losses['total_loss'])
		pickle.dump(RESULTS['LOSS'], open(RESULTS['results_dir'] + 'loss', 'wb'))
		RESULTS['BCELOSS'].append(losses['BCEloss'])
		pickle.dump(RESULTS['BCELOSS'], open(RESULTS['results_dir'] + 'BCEloss', 'wb'))
		RESULTS['AUTOENCODER_LOSS'].append(losses['AutoEncoder_loss'])
		pickle.dump(RESULTS['AUTOENCODER_LOSS'], open(RESULTS['results_dir'] + 'AutoEncoder_loss', 'wb'))
		# RESULTS['teachLOSS'].append(losses['teachloss'])
		# pickle.dump(RESULTS['teachLOSS'], open(RESULTS['results_dir'] + 'teachloss', 'wb'))
		# RESULTS['crossLOSS'].append(losses['crossloss'])
		# pickle.dump(RESULTS['crossLOSS'], open(RESULTS['results_dir'] + 'crossloss', 'wb'))
		# RESULTS['RegLOSS'].append(losses['Regloss'])
		# pickle.dump(RESULTS['RegLOSS'], open(RESULTS['results_dir'] + 'Regloss', 'wb'))
		RESULTS['ACC'].append(metrics['ACC'])
		pickle.dump(RESULTS['ACC'], open(RESULTS['results_dir'] + 'ACC', 'wb'))
		RESULTS['mAP'].append(metrics['meanAP'])
		pickle.dump(RESULTS['mAP'], open(RESULTS['results_dir'] + 'mAP', 'wb'))
		RESULTS['miF1'].append(metrics['miF1'])
		pickle.dump(RESULTS['miF1'], open(RESULTS['results_dir'] + 'miF1', 'wb'))
		RESULTS['maF1'].append(metrics['maF1'])
		pickle.dump(RESULTS['maF1'], open(RESULTS['results_dir'] + 'maF1', 'wb'))
		RESULTS['Epoch'].append(epoch)
		pickle.dump(RESULTS['Epoch'], open(RESULTS['results_dir'] + 'Epoch', 'wb'))
		pickle.dump(allAP, open(RESULTS['results_dir'] + 'all_AP', 'wb'))
		vis.line(np.array(RESULTS['LOSS']),np.array(RESULTS['Epoch']),win='LOSS',name='LOSS',opts=dict(title='TotalLOSS'))
		vis.line(np.array(RESULTS['BCELOSS']),np.array(RESULTS['Epoch']),win='BCELOSS',name='LOSS',opts=dict(title='BCELOSS'))
		# vis.line(np.array(RESULTS['teachLOSS']),np.array(RESULTS['Epoch']),win='teachLOSS',name='LOSS',opts=dict(title='teachLOSS'))
		vis.line(np.array(RESULTS['AUTOENCODER_LOSS']),np.array(RESULTS['Epoch']),win='AutoEncoder_loss',name='LOSS',opts=dict(title='AutoEncoder_loss'))
		# vis.line(np.array(RESULTS['crossLOSS']),np.array(RESULTS['Epoch']),win='crossLOSS',name='LOSS',opts=dict(title='crossLOSS'))
		# vis.line(np.array(RESULTS['RegLOSS']),np.array(RESULTS['Epoch']),win='RegLOSS',name='LOSS',opts=dict(title='RegLOSS'))
		vis.line(np.array(RESULTS['ACC']),np.array(RESULTS['Epoch']),win='ACC',name='ACC',opts=dict(title='ACC'))
		vis.line(np.array(RESULTS['mAP']),np.array(RESULTS['Epoch']),win='mAP',name='mAP',opts=dict(title='mAP'))
		vis.line(np.array(RESULTS['miF1']),np.array(RESULTS['Epoch']), win='miF1', name='miF1',opts=dict(title='miF1'))
		vis.line(np.array(RESULTS['maF1']),np.array(RESULTS['Epoch']), win='maF1', name='maF1',opts=dict(title='maF1'))
		vis.bar(allAP,win='allAP',opts=dict(title='allAP'))
	if type=='val':
		RESULTS['ACC_val'].append(metrics['ACC'])
		pickle.dump(RESULTS['ACC_val'], open(RESULTS['results_dir'] + 'ACC_val', 'wb'))
		RESULTS['mAP_val'].append(metrics['meanAP'])
		pickle.dump(RESULTS['mAP_val'], open(RESULTS['results_dir'] + 'mAP_val', 'wb'))
		RESULTS['miF1_val'].append(metrics['miF1'])
		pickle.dump(RESULTS['miF1_val'], open(RESULTS['results_dir'] + 'miF1_val', 'wb'))
		RESULTS['maF1_val'].append(metrics['maF1'])
		pickle.dump(RESULTS['maF1_val'], open(RESULTS['results_dir'] + 'maF1_val', 'wb'))
		pickle.dump(allAP, open(RESULTS['results_dir'] + 'all_AP_val', 'wb'))
		vis.line(np.array(RESULTS['ACC_val']),np.array(RESULTS['Epoch']),win='ACC_val',name='ACC',opts=dict(title='ACC_val'))
		vis.line(np.array(RESULTS['mAP_val']),np.array(RESULTS['Epoch']),win='mAP_val',name='mAP',opts=dict(title='mAP_val'))
		vis.line(np.array(RESULTS['miF1_val']),np.array(RESULTS['Epoch']), win='miF1_val', name='miF1',opts=dict(title='miF1_val'))
		vis.line(np.array(RESULTS['maF1_val']),np.array(RESULTS['Epoch']), win='maF1_val', name='maF1',opts=dict(title='maF1_val'))
		vis.bar(allAP,win='allAP_val',opts=dict(title='allAP_val'))

	return RESULTS