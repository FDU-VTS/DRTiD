import numpy as np
import torch
import argparse
import os
import torch.nn as nn
from torch import optim
from importlib import import_module
from torch.utils.data import DataLoader
from dataset_op import deepdrid_clf, drtid
from datetime import datetime
from utils.functions import progress_bar
from torchnet import meter
import torch.nn.functional as F
from sklearn.metrics import f1_score,roc_auc_score,cohen_kappa_score,accuracy_score,confusion_matrix
from resnet.resnet_crossfit import resnet50 as resnet50_crossfit
from utils.lr_scheduler import LRScheduler
import random
import torch.backends.cudnn as cudnn


import warnings
warnings.filterwarnings('ignore')

from thop import profile

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('--visname', '-vis', default='kaggle', help='visname')
parser.add_argument('--batch-size', '-bs', default=64, type=int, help='batch-size')
parser.add_argument('--lr', '-lr', default=1e-3, type=float, help='lr')
parser.add_argument('--epochs', '-eps', default=100, type=int, help='epochs')
parser.add_argument('--n_classes', '-n-cls', default=5, type=int, help='n-classes')
parser.add_argument('--pretrained', '-pre', default=False, type=bool, help='use pretrained model')
parser.add_argument('--dataset', '-data', default='drtid', type=str, help='dataset')

parser.add_argument('--xfmer_hidden_size', '-xhs', default=1024, type=int, help='xfmer hidden size')
parser.add_argument('--xfmer_layer', '-xl', default=2, type=int, help='xfmer layer')
parser.add_argument('--pool', '-pool', default='avg', type=str, help='pool: avg / max')
parser.add_argument('--p_threshold', '-p', default=0.5, type=float, help='p threshold')

parser.add_argument("--lr_mode", default="cosine", type=str)
parser.add_argument("--warmup_epochs", default=0, type=int)
parser.add_argument("--warmup_lr", default=0.0, type=float)
parser.add_argument("--targetlr", default=0.0, type=float)
parser.add_argument("--lambda_value", default=0.25, type=float)

val_epoch = 1
test_epoch = 1


my_whole_seed = 0
torch.manual_seed(my_whole_seed)
torch.cuda.manual_seed_all(my_whole_seed)
torch.cuda.manual_seed(my_whole_seed)
np.random.seed(my_whole_seed)
random.seed(my_whole_seed)
cudnn.deterministic = True
cudnn.benchmark = False 
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)/1048576

def parse_args():
    global args
    args = parser.parse_args()


best_acc = 0
best_kappa = 0

best_test_acc = 0
best_test_kappa = 0


def main_train():
    
    global best_kappa
    global save_dir
    
    parse_args()

    net = resnet50_crossfit(pretrained=False,num_classes=5,xfmer_hidden_size=args.xfmer_hidden_size,xfmer_layer=args.xfmer_layer,pool=args.pool,p_threshold=args.p_threshold)
    print(net)
    print(count_parameters(net))
    if args.pretrained:
        print ("==> Load pretrained model")
        ckpt = torch.load('./pretrained/kaggle_res50.pkl')
        state_dict = ckpt['net']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace("module.", "")
            unParalled_state_dict[new_key] = state_dict[key]
        net.load_state_dict(unParalled_state_dict,False)

    net = nn.DataParallel(net)
    net = net.cuda()

    if args.dataset == 'drtid':
        trainset = drtid(train=True, test=False)
        valset = drtid(train=False, test=True)
    elif args.dataset == 'deepdrid':
        trainset = deepdrid_clf(train=True, val=False, test=False)
        valset = deepdrid_clf(train=False, val=True, test=False)

    trainloader = DataLoader(trainset,  shuffle=True, batch_size=args.batch_size, num_workers=4,pin_memory=True)
    valloader = DataLoader(valset, shuffle=False, batch_size=args.batch_size,num_workers=4,pin_memory=True)


    # optim & crit
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5) #1e-5
    lr_scheduler = LRScheduler(optimizer, len(trainloader), args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    con_matx = meter.ConfusionMeter(args.n_classes)

    save_dir = './checkpoints/' + args.visname + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_log=open('./logs/'+args.visname+'.txt','w')   

    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        con_matx.reset()
        net.train()
        total_loss = .0
        total = .0
        correct = .0
        count = .0

        for i, (x1, x2, label, img1_grid, id) in enumerate(trainloader):
            lr = lr_scheduler.update(i, epoch)

            x1 = x1.float().cuda()
            x2 = x2.float().cuda()
            img1_grid = img1_grid.cuda()
            label = label.cuda()
            y_pred = net(x1,x2,img1_grid)
            
            loss = criterion(y_pred, label)
            prediction = y_pred.max(1)[1]

            total_loss += loss.item()
            total += x1.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            con_matx.add(prediction.detach(),label.detach())
            correct += prediction.eq(label).sum().item()

            count += 1           
            
            progress_bar(i, len(trainloader), 'Loss: %.3f | Acc: %.3f '
                         % (total_loss / (i + 1), 100. * correct / total))
           
        test_log.write('Epoch:%d  lr:%.5f  Loss:%.4f  Acc:%.4f \n'%(epoch, lr, total_loss / count, correct/total))
        test_log.flush()  

        if (epoch+1)%val_epoch == 0:
            main_val(net, valloader, epoch, test_log)

@torch.no_grad()
def main_val(net, valloader, epoch, test_log):
    global best_acc
    global best_kappa

    net = net.eval()
    con_matx = meter.ConfusionMeter(args.n_classes)

    pred_list = []
    predicted_list = []
    label_list = []

    for i, (x1, x2, label, img1_grid, id) in enumerate(valloader):
        x1 = x1.float().cuda()
        x2 = x2.float().cuda()
        img1_grid = img1_grid.cuda()
        label = label.cuda()

        y_pred = net(x1,x2,img1_grid)
        con_matx.add(y_pred.detach(),label.detach())

        label_list.extend(label.cpu().detach())
        predicted_list.extend(torch.softmax(y_pred,dim=-1).squeeze(-1).cpu().detach().numpy())

        pred = y_pred.max(1)[1]
        pred_list.extend(pred.cpu().detach())

        progress_bar(i, len(valloader))

    kappa = cohen_kappa_score(np.array(label_list), np.array(pred_list), weights='quadratic')
    acc = accuracy_score(np.array(label_list), np.array(pred_list))
    auc = roc_auc_score(np.array(label_list), np.array(predicted_list), average='macro', multi_class='ovr')
    print('val epoch:', epoch, ' acc: ', acc, 'kappa: ', kappa, 'auc: ', auc, 'con: ', str(con_matx.value()))
    test_log.write('Val Epoch:%d   Accuracy:%.4f   kappa:%.4f   auc:%.4f  con:%s \n'%(epoch,acc, kappa, auc, str(con_matx.value())))
    test_log.flush()

    if kappa > best_kappa:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'kappa': kappa,
            'epoch': epoch,
        }
        save_name = os.path.join(save_dir, str(epoch) + '.pkl')
        torch.save(state, save_name)
        best_kappa = kappa


def Sensitivity(Y_test,Y_pred,n):#n为分类数
    
    sen = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)
        
    return sen

def Precise(Y_test,Y_pred,n):
    
    pre = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fp = np.sum(con_mat[:,i]) - tp
        pre1 = tp / (tp + fp)
        pre.append(pre1)
        
    return pre

def Specificity(Y_test,Y_pred,n):
    
    spe = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)
    
    return spe

def ACC(Y_test,Y_pred,n):
    
    acc = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        acc1 = (tp + tn) / number
        acc.append(acc1)
        
    return acc



if __name__ == '__main__':
    parse_args()
    main_train()
