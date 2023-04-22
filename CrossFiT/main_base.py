import numpy as np
import torch
import argparse
import os
import torch.nn as nn
from torch import optim
from importlib import import_module
from torch.utils.data import DataLoader
from dataset import deepdrid_clf, drtid
from datetime import datetime
from utils.functions import progress_bar
from torchnet import meter
import torch.nn.functional as F
from sklearn.metrics import f1_score,roc_auc_score,cohen_kappa_score,accuracy_score,confusion_matrix
from resnet.resnet import resnet50
from resnet.resnet_fusion2 import resnet50 as resnet50_fusion2
from utils.lr_scheduler import LRScheduler
import random
import torch.backends.cudnn as cudnn


TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('--visname', '-vis', default='kaggle', help='visname')
parser.add_argument('--batch-size', '-bs', default=64, type=int, help='batch-size')
parser.add_argument('--lr', '-lr', default=1e-3, type=float, help='lr')
parser.add_argument('--epochs', '-eps', default=100, type=int, help='epochs')
parser.add_argument('--n_classes', '-n-cls', default=5, type=int, help='n-classes')
parser.add_argument('--pretrained', '-pre', default=False, type=bool, help='use pretrained model')
parser.add_argument('--dataset', '-data', default='drtid', type=str, help='dataset')
parser.add_argument('--test', '-test', default=False, type=bool, help='test mode')
parser.add_argument('--fusion_category', '-fc', default='single', type=str, help='single/fusion2/fusion3')
parser.add_argument('--fusion_type', '-ft', default='1', type=str, help='single:1,2,3/fusion2:add,cat,avg,max/fusion3:avg,max')

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


def parse_args():
    global args
    args = parser.parse_args()


best_acc = 0
best_kappa = 0

best_test_acc = 0
best_test_kappa = 0


def main_single():
    
    global best_kappa
    global save_dir
    parse_args()

    net = resnet50(pretrained=False,num_classes=5)
    print(net)

    if args.pretrained:
        print ("==> Load pretrained model")
        ckpt = torch.load('./pretrained/kaggle_res50.pkl')
        state_dict = ckpt['net']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        net.load_state_dict(unParalled_state_dict,True)

    net = nn.DataParallel(net)
    net = net.cuda()

    if args.dataset == 'drtid':
        trainset = drtid(train=True, test=False)
        valset = drtid(train=False, test=True)
    elif args.dataset == 'deepdrid':
        trainset = deepdrid_clf(train=True, val=False, test=False)
        valset = deepdrid_clf(train=False, val=True, test=False)

    trainloader = DataLoader(trainset,  shuffle=True, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    valloader = DataLoader(valset, shuffle=False, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # optim & crit
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5) #1e-5
    lr_scheduler = LRScheduler(optimizer, len(trainloader), args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    save_dir = './checkpoints/' + args.visname + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_log=open('./logs/'+args.visname+'.txt','a')   

    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        net.train()
        total_loss = .0
        total = .0
        correct = .0
        count = .0

        for i, (x1, x2, label, id) in enumerate(trainloader):
            lr = lr_scheduler.update(i, epoch)

            x1 = x1.float().cuda()
            x2 = x2.float().cuda()
            label = label.cuda()

            if args.fusion_type == '1':
                y_pred = net(x1)
            elif args.fusion_type == '2':
                y_pred = net(x2)
            elif args.fusion_type == '3':
                y_pred1 = net(x1)
                y_pred2 = net(x2)          
            
            if args.fusion_type == '1' or args.fusion_type == '2':
                loss = criterion(y_pred, label)
                prediction = y_pred.max(1)[1]
            
                total_loss += loss.item()
                total += x1.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                correct += prediction.eq(label).sum().item()

                count += 1           
                progress_bar(i, len(trainloader), 'Loss: %.3f | Acc: %.3f '
                             % (total_loss / (i + 1), 100. * correct / total))
           
            elif args.fusion_type == '3':
                loss = criterion(y_pred1, label)+criterion(y_pred2, label)
                prediction1 = y_pred1.max(1)[1]  
                prediction2 = y_pred2.max(1)[1]                

                total_loss += loss.item()
                total += 2*x1.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                correct += prediction1.eq(label).sum().item()
                correct += prediction2.eq(label).sum().item()

                count += 2         
                
                progress_bar(i, len(trainloader), 'Loss: %.3f | Acc: %.3f '
                            % (total_loss / (2*(i + 1)), 100. * correct / total))
           
        test_log.write('Epoch:%d  lr:%.5f  Loss:%.4f  Acc:%.4f \n'%(epoch, lr, total_loss / count, correct/total))
        test_log.flush()  

        if (epoch+1)%val_epoch == 0:
            main_single_val(net, valloader, epoch, test_log)

@torch.no_grad()
def main_single_val(net, valloader, epoch, test_log):
    global best_acc
    global best_kappa

    net = net.eval()
    con_matx = meter.ConfusionMeter(args.n_classes)

    pred_list = []
    predicted_list = []
    label_list = []

    for i, (x1, x2, label, id) in enumerate(valloader):
        x1 = x1.float().cuda()
        x2 = x2.float().cuda()
        label = label.cuda()

        if args.fusion_type == '1':
            y_pred = net(x1)
            con_matx.add(y_pred.detach(),label.detach())

        elif args.fusion_type == '2':
            y_pred = net(x2)
            con_matx.add(y_pred.detach(),label.detach())

        elif args.fusion_type == '3':
            y_pred1 = net(x1)
            y_pred2 = net(x2)  
            con_matx.add(y_pred1.detach(),label.detach())
            con_matx.add(y_pred2.detach(),label.detach())

        if args.fusion_type == '1' or args.fusion_type == '2':
            predicted_list.extend(y_pred.squeeze(-1).cpu().detach())
            label_list.extend(label.cpu().detach())
            pred = y_pred.max(1)[1]
            pred_list.extend(pred.cpu().detach())

        if args.fusion_type =='3':
            label_list.extend(label.cpu().detach())
            label_list.extend(label.cpu().detach())
            predicted_list.extend(y_pred1.squeeze(-1).cpu().detach())
            predicted_list.extend(y_pred2.squeeze(-1).cpu().detach())

            pred1 = y_pred1.max(1)[1]
            pred2 = y_pred2.max(1)[1]
            pred_list.extend(pred1.cpu().detach())
            pred_list.extend(pred2.cpu().detach())

        progress_bar(i, len(valloader))

    kappa = cohen_kappa_score(np.array(label_list), np.array(pred_list), weights='quadratic')
    acc = accuracy_score(np.array(label_list), np.array(pred_list))
    print('val epoch:', epoch, ' acc: ', acc, 'kappa: ', kappa, 'con: ', str(con_matx.value()))
    test_log.write('Val Epoch:%d   Accuracy:%.4f   kappa:%.4f  con:%s \n'%(epoch,acc, kappa, str(con_matx.value())))
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



def main_fusion2():
    
    global best_kappa
    global save_dir
    
    parse_args()

    net = resnet50_fusion2(pretrained=False,num_classes=5,fusion_type=args.fusion_type)

    if args.pretrained:
        print ("==> Load pretrained model")
        ckpt = torch.load('./pretrained/kaggle_res50.pkl')
        state_dict = ckpt['net']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        net.load_state_dict(unParalled_state_dict,False)

    net = nn.DataParallel(net)
    net = net.cuda()

    if args.dataset == 'drtid':
        trainset = drtid(train=True, test=False)
        valset = drtid(train=False, test=True)
    elif args.dataset == 'deepdrid':
        trainset = deepdrid_clf(train=True, val=False, test=False)
        valset = deepdrid_clf(train=False, val=True, test=False)

    trainloader = DataLoader(trainset,  shuffle=True, batch_size=args.batch_size, num_workers=4)
    valloader = DataLoader(valset, shuffle=False, batch_size=args.batch_size,num_workers=4)


    # optim & crit
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5) #1e-5
    lr_scheduler = LRScheduler(optimizer, len(trainloader), args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    con_matx = meter.ConfusionMeter(args.n_classes)

    save_dir = './checkpoints/' + args.visname + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_log=open('./logs/'+args.visname+'.txt','a')   

    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        con_matx.reset()
        net.train()
        total_loss = .0
        total = .0
        correct = .0
        count = .0

        for i, (x1, x2, label, id) in enumerate(trainloader):
            lr = lr_scheduler.update(i, epoch)

            x1 = x1.float().cuda()
            x2 = x2.float().cuda()
            label = label.cuda()
            y_pred = net(x1,x2)
            
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
           
        test_log.write('Epoch:%d  lr:%.5f  Loss:%.4f \n'%(epoch, lr, total_loss / count))
        test_log.flush()  

        if (epoch+1)%val_epoch == 0:
            main_fusion2_val(net, valloader, epoch, test_log)

@torch.no_grad()
def main_fusion2_val(net, valloader, epoch, test_log):
    global best_acc
    global best_kappa

    net = net.eval()
    con_matx = meter.ConfusionMeter(args.n_classes)

    pred_list = []
    predicted_list = []
    label_list = []

    for i, (x1, x2, label, id) in enumerate(valloader):
        x1 = x1.float().cuda()
        x2 = x2.float().cuda()
        label = label.cuda()

        y_pred = net(x1,x2)
        con_matx.add(y_pred.detach(),label.detach())

        label_list.extend(label.cpu().detach())
        predicted_list.extend(y_pred.squeeze(-1).cpu().detach())

        pred = y_pred.max(1)[1]
        pred_list.extend(pred.cpu().detach())

        progress_bar(i, len(valloader))

    kappa = cohen_kappa_score(np.array(label_list), np.array(pred_list), weights='quadratic')
    acc = accuracy_score(np.array(label_list), np.array(pred_list))
    print('val epoch:', epoch, ' acc: ', acc, 'kappa: ', kappa, 'con: ', str(con_matx.value()))
    test_log.write('Val Epoch:%d   Accuracy:%.4f   kappa:%.4f  con:%s \n'%(epoch,acc, kappa, str(con_matx.value())))
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



def main_fusion3():
    
    global best_kappa
    global save_dir
    
    parse_args()

    net = resnet50(pretrained=False,num_classes=5)
    print(net)

    if args.pretrained:
        print ("==> Load pretrained model")
        ckpt = torch.load('./pretrained/kaggle_res50.pkl')
        state_dict = ckpt['net']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        net.load_state_dict(unParalled_state_dict,True)

    net = nn.DataParallel(net)
    net = net.cuda()

    if args.dataset == 'drtid':
        trainset = drtid(train=True, test=False)
        valset = drtid(train=False, test=True)
    elif args.dataset == 'deepdrid':
        trainset = deepdrid_clf(train=True, val=False, test=False)
        valset = deepdrid_clf(train=False, val=True, test=False)

    trainloader = DataLoader(trainset,  shuffle=True, batch_size=args.batch_size, num_workers=4)
    valloader = DataLoader(valset, shuffle=False, batch_size=args.batch_size,num_workers=4)

    # optim & crit
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5) #1e-5
    lr_scheduler = LRScheduler(optimizer, len(trainloader), args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    con_matx = meter.ConfusionMeter(args.n_classes)

    save_dir = './checkpoints/' + args.visname + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_log=open('./logs/'+args.visname+'.txt','a')   

    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        con_matx.reset()
        net.train()
        total_loss = .0
        total = .0
        correct = .0
        count = .0

        for i, (x1, x2, label, id) in enumerate(trainloader):
            lr = lr_scheduler.update(i, epoch)

            x1 = x1.float().cuda()
            x2 = x2.float().cuda()
            label = label.cuda()

            y_pred1 = net(x1)
            y_pred2 = net(x2)            
            
            loss = criterion(y_pred1, label)+criterion(y_pred2, label)
            prediction1 = y_pred1.max(1)[1]  
            prediction2 = y_pred2.max(1)[1]                

            total_loss += loss.item()
            total += 2*x1.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # con_matx.add(prediction.detach(),label.detach())
            correct += prediction1.eq(label).sum().item()
            correct += prediction2.eq(label).sum().item()

            count += 1           
            
            progress_bar(i, len(trainloader), 'Loss: %.3f | Acc: %.3f '
                        % (total_loss / (2*(i + 1)), 100. * correct / total))
            
        test_log.write('Epoch:%d  lr:%.5f  Loss:%.4f \n'%(epoch, lr, total_loss / count))
        test_log.flush()  
        if (epoch+1)%val_epoch == 0:
            main_fusion3_val(net, valloader, epoch, test_log)


@torch.no_grad()
def main_fusion3_val(net, valloader, epoch, test_log):
    global best_acc
    global best_kappa

    net = net.eval()
    con_matx = meter.ConfusionMeter(args.n_classes)

    pred_list = []
    predicted_list =[]
    label_list = []

    for i, (x1,x2,label,id) in enumerate(valloader):
        x1 = x1.float().cuda()
        x2 = x2.float().cuda()
        label = label.cuda()

        y1_pred = net(x1)
        y2_pred = net(x2)

        if args.fusion_type == 'max':
            pred = torch.zeros(x1.size(0)).long().cuda() #bs
            pred1 = y1_pred.max(1)[1] #bs
            pred2 = y2_pred.max(1)[1]

            for bs in range(x1.size(0)):
                if pred1[bs] >= pred2[bs]:
                    pred[bs] = pred1[bs]
                else:
                    pred[bs] = pred2[bs] 

        elif args.fusion_type == 'avg':
            y1_pred = F.softmax(y1_pred,dim=1)
            y2_pred = F.softmax(y2_pred,dim=1)
            y_pred = 0.5*(y1_pred+y2_pred)
            pred = y_pred.max(1)[1]

        pred_list.extend(pred.cpu().detach())
        label_list.extend(label.cpu().detach())

    kappa = cohen_kappa_score(np.array(label_list), np.array(pred_list), weights='quadratic')
    acc = accuracy_score(np.array(label_list), np.array(pred_list))
    print('val epoch:', epoch, ' acc: ', acc, 'kappa: ', kappa)
    test_log.write('Val Epoch:%d   Accuracy:%.4f   kappa:%.4f  \n'%(epoch,acc, kappa))
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
        
    return acc, con_mat

if __name__ == '__main__':
    parse_args()
    if args.fusion_category == 'single':
        main_single()
    elif args.fusion_category == 'fusion2':
        main_fusion2()
    elif args.fusion_category == 'fusion3':
        main_fusion3()
