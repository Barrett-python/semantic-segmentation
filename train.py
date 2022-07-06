import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import OCT_Dataset
from evalution_segmentaion import eval_semantic_segmentation
from DaTransNet import DaTransNet
import cfg
from tqdm import tqdm
import numpy as np



def val(model,epoch):
    net = model.eval()   
    train_loss = 0
    train_acc = 0
    train_miou = 0
    train_class_acc = 0
    for i, sample in tqdm(enumerate(val_data)):
        img_data = Variable(sample['img'].to(device))   # [16, 3, 512, 512]
        img_label = Variable(sample['label'].to(device)).squeeze(dim=1)    # [16, 512, 512]
        out_0_0, out_1_1 = net(img_data)     # [16, 4, 512, 512]
        out = out_0_0 + out_1_1
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
#        print(np.unique(pre_label))
        pre_label = [i for i in pre_label]
        true_label = img_label.data.cpu().numpy()
#        print(np.unique(true_label))
        true_label = [i for i in true_label]
        eval_metrix = eval_semantic_segmentation(pre_label, true_label)
        train_acc += eval_metrix['mean_class_accuracy']
        train_miou += eval_metrix['miou']
        train_class_acc += eval_metrix['class_accuracy']
            
    with open('/mnt/DATA-1/DATA-2/Feilong/sematic_segmentation/recoder.txt','a') as f:
        f.write('|Train Acc|: {:.5f}|Train Mean IU|: {:.5f}\n|Train_class_acc|:{:}\n'.format(
        train_acc / len(train_data),
        train_miou / len(train_data),
        train_class_acc / len(train_data)))

    if max(best) <= train_miou / len(train_data):
        best.append(train_miou / len(train_data))
        t.save(net.state_dict(), './weight/{}.pth'.format(epoch))
        

def train(model):
    best = [0]
    for epoch in range(cfg.EPOCH_NUMBER):
        net = model.train()
        for i, sample in tqdm(enumerate(train_data)):
            img_data = Variable(sample['img'].to(device))  
            img_label = Variable(sample['label'].to(device)).squeeze(dim=1)
            img_label = torch.tensor(img_label, dtype=torch.long) 
            out_0, out_1 = net(img_data) 
            
            loss_0 = criterion(out_0, img_label)
            loss_1 = criterion(out_1, img_label)
            loss = loss_0 + loss_1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 评估
        val(model, epoch)

if __name__ == "__main__":

    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    OCT_train = OCT_Dataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
    OCT_val = OCT_Dataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)
    train_data = DataLoader(OCT_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=8)
    val_data = DataLoader(OCT_val, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=8)
    model = DaTransNet().to(device)
    criterion = nn.NLLLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best = [0]
    train(model)

