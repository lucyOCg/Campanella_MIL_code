import sys
import os
import numpy as np
import argparse
import random
import openslide
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import wandb
from matplotlib import pyplot
from numpy import sqrt
from numpy import argmax

wandb.login()


#test comment :] version 2


wandb.init(project="my-test-project", entity="lucyocg")

#wandb.config = {
#"learning_rate": 0.0001,
#"epochs": 12,
#"batch_size": 512}

#wandb.init(config={"lr": 0.0001})
#wandb.config.epochs = 12
#wandb.config.batch_size = 512
#wandb.init(config={"lr": 0.1})
#wandb.config.epochs = 4

#parser = argparse.ArgumentParser()
#parser.add_argument('-b', '--batch-size', type=int, default=512, metavar='N',
                     #help='input batch size for training (default: 512)')
#args = parser.parse_args()
#wandb.config.update(args) # adds all of the arguments as config variables

hyperparameter_defaults = dict(
    epochs=12,
    batch_size=512,
    learning_rate=0.0001,
    )

config_dictionary = dict(
    params=hyperparameter_defaults,
    )

wandb.init(config=config_dictionary)



parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--train_lib', type=str, default='', help='path to train MIL library binary')
parser.add_argument('--val_lib', type=str, default='', help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='.', help='name of output file')
parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=1, type=int, help='test on val every (default: 10)')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--k', default=1, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')

best_acc = 0
def main():
    global args, best_acc
    args = parser.parse_args()

    #cnn
    model = models.resnet34(True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      model = nn.DataParallel(model)#, device_ids = [0, 1]) #added
      #torch.cuda.set_device('cuda:0')
      #model.to(f'cuda:{model.device_ids[0]}')#added
    model.cuda()

    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights,args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    cudnn.benchmark = True


    #normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    #load data
    train_dset = MILdataset(args.train_lib, trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    if args.val_lib:
        val_dset = MILdataset(args.val_lib, trans)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

    #open output file
    fconv = open(os.path.join(args.output,'convergence_retrain.csv'), 'w')
    fconv.write('epoch,metric,value\n')
    fconv.close()

    #model = model(*args, **kwargs)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = nn.DataParallel(model)
    ckp_path = "/nobackup/projects/bdlds05/lucyg/MIL-nature-medicine-2019-master/results/model2_12epochs/checkpoint_best.pth"
    #model.load_state_dict(torch.load(ckp_path), strict=False)#added
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    #loss = checkpoint['loss']

    #model, optimizer, start_epoch = load_ckp(ckp_path, model, optimizer)

    
    #loop through epochs
    #train_loss = []
    #val_loss_1 = []
    save_dir = "/nobackup/projects/bdlds05/lucyg/MIL-nature-medicine-2019-master/metrics/retrained_mod/"
    for epoch in range(args.nepochs):
        train_dset.setmode(1)
        probs = inference(epoch, train_loader, model)
        maxs = group_max(np.array(train_dset.slideIDX), probs, len(train_dset.targets))
        train_pred = [1 if x >= 0.5 else 0 for x in maxs]
        err,fpr,fnr = calc_err(train_pred, train_dset.targets)
        trainy = np.asarray(train_dset.targets)
        #print('trainy:', len(trainy), trainy)
        train_yhat = np.asarray(maxs)
        #print('train_yhat:', len(train_yhat), train_yhat)
        fpr_1, tpr, thresholds = roc_curve(trainy, train_yhat)
        roc_score = roc_auc_score(trainy, train_yhat)
        gmeans = sqrt(tpr * (1-fpr_1))
        ix = argmax(gmeans)
        pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
        pyplot.plot(fpr_1, tpr, marker='.', label='Campanella MIL')
        pyplot.scatter(fpr_1[ix], tpr[ix], marker='o', color='black', label='Best')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.legend()
        # show the plot
        pyplot.savefig(save_dir+'train_gmean_roc_curve_'+ str(epoch +1) + '.png')
        pyplot.clf()
        data = confusion_matrix(trainy, np.asarray(train_pred))
        df_cm = pd.DataFrame(data, columns=np.unique(trainy), index = np.unique(trainy))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        tp = df_cm.iloc[1][1]
        fp = df_cm.iloc[0][1]
        fn = df_cm.iloc[1][0]
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = (2*precision*recall)/(precision+recall)
        #plt.figure(figsize = (10,7))
        sn.set(font_scale=1.0)#for label size
        sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 14}, fmt='g')# font size
        confusion_plot = sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 14}, fmt='g')
        fig = confusion_plot.get_figure()
        fig.savefig(os.path.join(save_dir+'ConfusionMatrix' + str(epoch+1)+ '.png')) 
        #fig.savefig('ConfusionMatrix.png')
        pyplot.clf() 
        
        #print('probs_train:', probs)
        topk = group_argtopk(np.array(train_dset.slideIDX), probs, args.k)
        train_dset.maketraindata(topk)
        train_dset.shuffletraindata()
        train_dset.setmode(2)
        loss = train(epoch, train_loader, model, criterion, optimizer)
        #train_loss.append(loss)
        metrics = {"train loss": loss,
                   "train G-mean": gmeans[ix],
                   "train best threshold": thresholds[ix],
                   "train error": err,
                   "train fpr": fpr,
                   "train fnr": fnr,
                   "train f1": f1,
                   "train ROC AUC score": roc_score,
                   "train precision": precision,
                   "train recall": recall}
        wandb.log(metrics)
        print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs, loss))
        fconv = open(os.path.join(args.output, 'convergence_retrain.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch+1,loss))
        fconv.close()

        #Validation
        print('args.val_lib', args.val_lib)
        print('(epoch+1) % args.test_every', (epoch+1) % args.test_every)
        if args.val_lib and (epoch+1) % args.test_every == 0:
            val_dset.setmode(1)
            probs = inference(epoch, val_loader, model)
            #print('len(probs)', len(probs))
            #print('probs_val:', probs)
            maxs = group_max(np.array(val_dset.slideIDX), probs, len(val_dset.targets))
            pred = [1 if x >= 0.5 else 0 for x in maxs]
            print('pred:', pred)
            err,fpr,fnr = calc_err(pred, val_dset.targets)
            topk = group_argtopk(np.array(val_dset.slideIDX), probs, args.k)
            val_dset.maketraindata(topk)
            val_dset.shuffletraindata()
            val_dset.setmode(2)
            loss = val_loss(val_loader, model, criterion)
            #val_loss.append(loss)
            testy = np.asarray(val_dset.targets)
            yhat = np.asarray(maxs)
            fpr_1, tpr, thresholds = roc_curve(testy, yhat)
            roc_score = roc_auc_score(testy, yhat)
            gmeans = sqrt(tpr * (1-fpr_1))
            ix = argmax(gmeans)
            data = confusion_matrix(testy, np.asarray(pred))
            df_cm = pd.DataFrame(data, columns=np.unique(testy), index = np.unique(testy))
            df_cm.index.name = 'Actual'
            df_cm.columns.name = 'Predicted'
            tp = df_cm.iloc[1][1]
            fp = df_cm.iloc[0][1]
            fn = df_cm.iloc[1][0]
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f1 = (2*precision*recall)/(precision+recall)
            #plt.figure(figsize = (10,7))
            sn.set(font_scale=1.0)#for label size
            sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 14}, fmt='g')# font size
            confusion_plot = sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 14}, fmt='g')
            fig = confusion_plot.get_figure()
            fig.savefig(os.path.join(save_dir+'val_ConfusionMatrix' + str(epoch+1)+ '.png'))
            pyplot.clf()
            val_metrics = {"epoch": epoch+1,
                           "validation error": err,
                           "validation fpr": fpr,
                           "validation fnr": fnr,
                           "Best threshold": thresholds[ix],
                           "validation G-mean": gmeans[ix],
                           "validation ROC AUC score": roc_score,
                           "validation loss": loss,
                           "validation f1 score": f1,
                           "validation precision": precision,
                           "validation recall": recall}
            wandb.log(val_metrics) 
            # plot the roc curve for the model
            pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
            pyplot.plot(fpr_1, tpr, marker='.', label='Campanella MIL')
            pyplot.scatter(fpr_1[ix], tpr[ix], marker='o', color='black', label='Best')
            # axis labels
            pyplot.xlabel('False Positive Rate')
            pyplot.ylabel('True Positive Rate')
            pyplot.legend()
            # show the plot=
            pyplot.savefig(save_dir+'val_gmean_roc_curve_'+ str(epoch +1) + '.png')
            pyplot.clf() 
            #pyplot.show()
            low_probs = 1 - yhat
            all_probs = np.transpose(np.asarray([low_probs, yhat]))
            print('all_probs', all_probs)
            labs = ['low immune', 'high immune']
            wandb.log({"roc_curve" : wandb.plot.roc_curve(testy, all_probs, labs)})
            print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, err, fpr, fnr))
            fconv = open(os.path.join(args.output, 'convergence_retrain.csv'), 'a')
            fconv.write('{},error,{}\n'.format(epoch+1, err))
            fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
            fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
            fconv.close()
            #Save best model
            err = (fpr+fnr)/2.
            if 1-err >= best_acc:
                best_acc = 1-err
                obj = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict()
                }
                torch.save(obj, os.path.join(args.output,'checkpoint_retrain.pth'))
  #epochs = [*range(0, len(val_loss), 1)]#1 len(val_loss)
  #pyplot.plot(epochs, val_loss, label = "val_loss")
  #pyplot.plot(epochs, train_loss, label = "train_loss")
  #pyplot.xlabel("Epochs")
  #pyplot.ylabel("Loss")
  #pyplot.legend()
  #pyplot.savefig('/metrics/loss'+ str(epoch +1) + '.png'  
  #pyplot.clf()


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath, strict=False)
    #model = nn.DataParallel(model)#added
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']        
     
    
          

def inference(run, loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(loader)))
            input = input.cuda()
            #input = input.to(f'cuda:{model.device_ids[0]}')#added
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy()

def train(run, loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        #input = input.to(f'cuda:{model.device_ids[0]}')#added
        target = target.cuda()
        #target = target.to(f'cuda:{model.device_ids[0]}')
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)
    return running_loss/len(loader.dataset)
    
def val_loss(loader, model, criterion):
    model.eval()
    running_loss = 0.
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        running_loss += loss.item()*input.size(0)
    return running_loss/len(loader.dataset)
    


def calc_err(pred,real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
    return err, fpr, fnr

def group_argtopk(groups, data,k=1):
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])

def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return out

class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None):
        lib = torch.load(libraryfile)
        slides = []
        for i,name in enumerate(lib['slides']):
            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i+1, len(lib['slides'])))
            sys.stdout.flush()
            slides.append(openslide.OpenSlide(name))
        
        print('')
        #Flatten grid
        grid = []
        slideIDX = []
        for i,g in enumerate(lib['grid']):
            grid.extend(g)
            slideIDX.extend([i]*len(g))
        #print('GRID', grid)
        #breakpoint()
        print('Number of tiles: {}'.format(len(grid)))
        self.slidenames = lib['slides']
        self.slides = slides
        self.targets = lib['targets']
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None
        self.mult = lib['mult']
        self.size = int(np.round(224*lib['mult']))
        self.level = lib['level']
    def setmode(self,mode):
        self.mode = mode
    def maketraindata(self, idxs):
        #print('x', len(x))
        self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]]) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def __getitem__(self,index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            coord = self.grid[index]
            img = self.slides[slideIDX].read_region(coord,self.level,(self.size,self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            slideIDX, coord, target = self.t_data[index]
            img = self.slides[slideIDX].read_region(coord,self.level,(self.size,self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target
    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)

if __name__ == '__main__':
    main()
