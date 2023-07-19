import os
import time
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import Model
from src.dataset import build_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

cfg = {}
cfg["device"] = "cuda:0" if torch.cuda.is_available() else "cpu:0"
cfg["batch_size"] = 1
cfg['image_dir'] = '/second_disk/Seed_Data_CY-150'
cfg['label_dir'] = '/second_disk/Seed_Data_CY-150/labels'
cfg['presave_path'] = 'data_22c_v1'

model_config = {
    "dims":[64,128,192,256],
    "depths":[2,2,6,2],
    "window_size":[1,1,1,1],
    "ks":[1,1,1,1],
    "num_attn":5,
    "num_classes":2
}
cfg['model'] = model_config


def ro_curve(y_preds, y_labels, figure_file, names):
    '''
        y_pred is a list of length n.  (0,1)
        y_label is a list of same length. 0/1
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py  
    '''
    
    line_width = 2
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.style'] = 'italic'
    fontdict = {'family': 'Times New Roman',
			'size': 12,
			'style': 'italic'
    }
    plt.plot([0, 1], [0, 1], color='navy', lw=line_width, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver Operating Characteristic Curve')
    
    colors = ['b','g','r','c']
    for y_pred,y_label,name,color in zip(y_preds,y_labels,names,colors):
        y_label = np.array(y_label)
        y_pred = np.array(y_pred)    
        print(y_label.shape,y_pred.shape)
        fpr, tpr, _ = roc_curve(y_label, y_pred)
        roc_auc = auc(fpr, tpr)
        _method_name = name + ' (area = %0.2f)' % roc_auc
        plt.plot(fpr, tpr,
            lw=line_width, label= _method_name,color=color)
    
    plt.legend(loc="lower right")
    plt.savefig("outputs/" + figure_file + ".jpeg",dpi=300,format='jpeg')
    return 

@torch.no_grad()
def eval_cls_acc(model,data_loader,device):
    count = 0
    top1_acc_count = 0
    tbar = tqdm(data_loader)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    y_pred = []
    for data in tbar:
        try:
            data['seed_image'] = data['seed_image'].to(cfg['device']).float()
            data['seed_label'] = data['seed_label'].to(cfg['device']).long()
            # data['seed_shape'] = data['seed_shape'].to(cfg['device']).float()
            # data['seed_light'] = data['seed_light'].to(cfg['device']).float()
            yhat = model(data)
            y_pred.append(torch.softmax(yhat,dim=-1)[:,1])
        except Exception as e:
            print(e)
        count += data['seed_label'].shape[0]
        _,indices = torch.topk(yhat,k=1,dim=-1)
        indices = indices.squeeze()
        mask = (data['seed_label'] - indices) == 0
        top1_acc_count += torch.sum(mask)
        tp += torch.sum(mask * indices)
        fp += torch.sum(~mask * indices)
        tn += torch.sum(mask * (1-indices))
        fn += torch.sum(~mask * (1-indices))
        tbar.set_description("acc: {:.4f},precision: {:.4f}, recall: {}".format(top1_acc_count/count,tp / (tp + fp), tp / (tp + fn)))
    return tp,fp,tn,fn,y_pred
    
def eval_func(cfg):
    _, dataset_eval = build_dataset(cfg['image_dir'], cfg['label_dir'], cfg['presave_path'],wants=['eval'])
    print(len(dataset_eval))
    eval_loaders = DataLoader(
        dataset_eval,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=8
    )
    weight_paths = [
        'weights/mbformer-bs64-tm-22c/model_8000_0.0196.pth'
    ]
    names = ['MsiFormer']
    models = [
        Model(cfg['model']).to(cfg['device'])
    ]
    y_label = np.array(dataset_eval.seed_labels[:,0],np.float32)
    y_preds = []
    for model,name,weight_path in zip(models,names,weight_paths):
        model.eval()
        ckpts = torch.load(weight_path,map_location=cfg['device'])
        model.load_state_dict(ckpts['model'])
        
        tp,fp,tn,fn,y_pred = eval_cls_acc(model, eval_loaders, cfg['device'])
        acc = (tp + tn) / (tp+fp+tn+fn)
        precision = tp / (tp + fp)
        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        tnr = tn / (tn + fp)
        fnr = fn / (fn + tp)
        cs = '{} tp:{:3d} tn:{:3d} fp:{:3d} fn:{:3d} acc:{:.4f} precision:{:.4f} tpr:{:.4f} fpr:{:.4f} tnr:{:.4f} fnr:{:.4f}\n'.format(
            name,tp,tn,fp,fn,acc,precision,tpr,fpr,tnr,fnr
        )
        print(cs)
        y_preds.append(torch.cat(y_pred,dim=0).cpu().numpy())
    
    y_preds = np.array(y_preds,dtype=np.float32)
    if len(y_label.shape) == 1:
        y_label = y_label[None,:]
    print(y_preds.shape)
    print(y_label.shape)
    np.save('outputs/y_preds_v2.npy',y_preds)
    np.save('outputs/y_label_v2.npy',y_label)
    ro_curve(y_preds, y_label, 'roc.jpeg',name)
        
if __name__ == "__main__":
    eval_func(cfg)