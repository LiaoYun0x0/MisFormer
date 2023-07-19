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
import argparse
import torchvision as tv

from src.model import Model
from src.dataset import build_dataset

cfg = {}
cfg["device"] = "cuda:0" if torch.cuda.is_available() else "cpu:0"
cfg['output_dim'] = 256
cfg["batch_size"] = 64
cfg["iterations"] = 10000
cfg["eval_steps"] = 1000
cfg["save_weights_steps"] = cfg['eval_steps']
cfg["init_lr"] = 1e-3
cfg["end_lr"] = 1e-5
cfg["cos_lr_t_max"] = cfg["iterations"]
cfg["weight_decay"] = 5e-4
cfg["weights_dir"] = "./weights/mbformer-bs64-tm-22c"
cfg["restore_weight_path"] = ""
cfg['image_dir'] = '/second_disk/Seed_Data_1.0'
cfg['label_dir'] = '/second_disk/Seed_Data_1.0/labels'
cfg['presave_path'] = 'data_22c_v1'
cfg['eval_log'] = 'eval_mbformer.log'

model_config = {
    "dims":[64,128,192,256],
    "depths":[2,2,6,2],
    "window_size":[1,1,1,1],
    "ks":[1,1,1,1],
    "num_attn":5,
    "num_classes":2,
    "in_chans":22
}
cfg['model'] = model_config

def save_weights(model,weight_dir,model_name,step,loss,optimizer,lr_scheduler):
    checkpoints = {
        'iter_step':step,
        "model":model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler":lr_scheduler.state_dict()
    }
    weight_path = '{}/{}_{}_{:.4f}.pth'.format(weight_dir,model_name,step,loss)
    torch.save(checkpoints, weight_path)
    return weight_path

@torch.no_grad()
def eval_cls_acc(model,data_loader,device):
    count = 0
    top1_acc_count = 0
    tbar = tqdm(data_loader)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for data in tbar:
        try:
            data['seed_image'] = data['seed_image'].to(cfg['device']).float()
            data['seed_label'] = data['seed_label'].to(cfg['device']).long()
            yhat = model(data)
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
    return tp,fp,tn,fn

def train(cfg):
    if not os.path.exists(cfg["weights_dir"]):
        os.makedirs(cfg["weights_dir"])

    model = Model(cfg['model']).to(cfg['device'])
    model.train()
    tnum = 0
    for v in list(model.parameters()):
        num = 1
        for s in v.size():
            num *= s
        tnum += num
    print('num_parameters:',tnum)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["init_lr"], weight_decay=cfg["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["cos_lr_t_max"], eta_min=cfg["end_lr"])

    dataset_train, dataset_eval = build_dataset(cfg['image_dir'], cfg['label_dir'], cfg['presave_path'],test_num=120)
    print("train samples: {}, test samples: {}".format(len(dataset_train),len(dataset_eval)))
    train_loaders = DataLoader(
        dataset_train,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=8
    )
    eval_loaders = DataLoader(
        dataset_eval,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=8
    )
    
    train_iter = iter(train_loaders)
    if os.path.exists(cfg["restore_weight_path"]):
        ckpts = torch.load(cfg["restore_weight_path"])
        model.load_state_dict(ckpts['model'])
        print('restore ckpt!')
        iter_step = ckpts['iter_step'] + 1
        optimizer.load_state_dict(ckpts["optimizer"])
        lr_scheduler.load_state_dict(ckpts["lr_scheduler"])
        print("restore optimizer and lr_scheduler")
    else:
        iter_step = 0

    total_steps = cfg["iterations"]
    remain_steps = total_steps - iter_step
    moving_mean_loss = 0
    momentum = 0.95
    
    for _ in range(remain_steps):
        start = time.time()
        try:
            data = next(train_iter)
        except:
            train_iter = iter(train_loaders)
            data = next(train_iter)
        data['seed_image'] = data['seed_image'].to(cfg['device']).float()
        data['seed_label'] = data['seed_label'].to(cfg['device']).long()
        data['seed_shape'] = data['seed_shape'].to(cfg['device']).float()
        data['seed_light'] = data['seed_light'].to(cfg['device']).float()
            
        loss = model.forward_train(data)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=10,norm_type=2)
        optimizer.step()
        lr_scheduler.step()

        loss_val = loss.item()
        moving_mean_loss = moving_mean_loss * momentum + loss_val * (1-momentum)
        
        
        if iter_step % 50 == 0:
            print("step:{}, loss:{:.3f}, movin_mean_loss: {:.3f}, lr: {:.10f}, cost_time:{:.3f}" \
                .format(iter_step,loss.item(), moving_mean_loss,lr_scheduler._last_lr[0],time.time() - start))
        
        if iter_step % cfg["save_weights_steps"] == 0:
            path = save_weights(model,cfg["weights_dir"],'model',iter_step,moving_mean_loss,optimizer,lr_scheduler)
            print("save_weights to %s"%path)
        
        if iter_step % cfg['eval_steps'] == 0:
            model.eval()
            tp,fp,tn,fn = eval_cls_acc(model, eval_loaders, cfg['device'])
            acc = (tp + tn) / (tp+fp+tn+fn)
            precision = tp / (tp + fp)
            tpr = tp / (tp + fn)
            fpr = fp / (tn + fp)
            tnr = tn / (tn + fp)
            fnr = fn / (fn + tp)
            cs = 'tp:{:3d} tn:{:3d} fp:{:3d} fn:{:3d} acc:{:.4f} precision:{:.4f} tpr:{:.4f} fpr:{:.4f} tnr:{:.4f} fnr:{:.4f}\n'.format(
                tp,tn,fp,fn,acc,precision,tpr,fpr,tnr,fnr
            )
            with open(cfg['eval_log'],'a') as f:
                f.write(cs)
            model.train()
            
        iter_step += 1
        
    path = save_weights(model,cfg["weights_dir"],'final_model',iter_step,moving_mean_loss,using_optimizer,using_scheduler)
    print("save_weights to %s"%path)
    print("training process finished!")
    model.eval()
    acc1 = eval_cls_acc(model, eval_loaders, cfg['device'])
    print("acc1: {:.4f}".format(acc1))
    cs = '{} step: {} acc1: {:.4f} \n'.format(
        cfg['weights_dir'],iter_step,acc1)
    
def random_seed():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
        
if __name__ == "__main__":
    random_seed()
    train(cfg)