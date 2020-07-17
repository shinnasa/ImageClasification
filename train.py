# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 20:08:29 2020

@author: shinpei
"""

#IMPORTS
from torchvision import datasets, transforms, models
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
import argparse

#Default Arguments
IN_SIZE=25088
OUT_SIZE=102
BATCH_SIZE=64
DFLT_HIDD_SIZE=512
DFLT_LR=0.001
DFLT_EPOCHS=5
DFLT_DATA_DIR="/home/workspace/ImageClassifier"
DFLT_GPU=True
DFLT_NET="vgg16"
DFLT_SAVE_CHKPT="/home/workspace/ImageClassifier"
def parse_args():
    parser = argparse.ArgumentParser(description = "Training")
    parser.add_argument("--data_dir",type=str,default=DFLT_DATA_DIR,help="Path")
    parser.add_argument("--hidden_size",type=int,default=DFLT_HIDD_SIZE,help="Path")
    parser.add_argument("--LR",type=float,default=DFLT_LR,help="Path")
    parser.add_argument("--gpu",type=bool,default=DFLT_GPU,help="Path")
    parser.add_argument("--epochs",type=int,default=DFLT_EPOCHS,help="Path")
    parser.add_argument("--net",type=str,default=DFLT_NET,help="Path")
    parser.add_argument("--save_chkpt",type=str,default=DFLT_SAVE_CHKPT,help="Path")
    args=parser.parse_args()
    return args.data_dir,args.hidden_size,args.LR,args.epochs,args.gpu,args.net,args.save_chkpt
def get_data_loader(data_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.RandomResizedCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    # transforms.Normalize((0.5,0.5), (0.5,0.5)),

    # TODO: Load the datasets with ImageFolder
    train_dir=data_dir+"/flowers/train"
    valid_dir=data_dir+"/flowers/valid"
    test_dir=data_dir+"/flowers/test"
    train = datasets.ImageFolder(train_dir,transform=train_transforms)
    valid = datasets.ImageFolder(valid_dir,transform=test_transforms)
    test = datasets.ImageFolder(test_dir,transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_load = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    valid_load = torch.utils.data.DataLoader(valid, batch_size=64)
    test_load = torch.utils.data.DataLoader(test, batch_size=64)
    class2idx=train.class_to_idx
    return train_load,valid_load,test_load,class2idx



def save_checkpt(model,IN_SIZE,OUT_SIZE,hidden_size,LR,BATCH_SIZE,epochs,optimizer,net,save_chkpt):
    checkpoint = {"net":net,
                  'input_size': IN_SIZE,
                  'output_size':OUT_SIZE,
                  'hidden_size': hidden_size,
                  'learning_rate': LR,       
                  'batch_size': BATCH_SIZE,
                  'classifier' :model.classifier,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, save_chkpt+'/checkpoint.pth')

def train(data_dir,hidden_size,LR,epochs,gpu,net,save_chkpt):
    train_load,valid_load,test_load,class2idx=get_data_loader(data_dir)
    vgg16 =  getattr(models, net)(pretrained = True)
    output_size=OUT_SIZE
    input_size=IN_SIZE
    #224*224
    for param in vgg16.parameters():
        param.requires_grad = False
    
    vgg16.classifier=nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_size)),
                          ('drop', nn.Dropout(p = 0.5)),
                          ('relu1', nn.ReLU()),
                          ('output', nn.Linear(hidden_size, output_size)),
                          ('softmax', nn.LogSoftmax(dim=1))]))
    #
    criterion=nn.NLLLoss()
    optimizer = optim.Adam(vgg16.classifier.parameters(), lr = LR)

    vgg16.class_to_idx =class2idx
    batch_size=BATCH_SIZE
    if gpu==True:
        vgg16.cuda()
    for e in range(epochs):
        running_loss = 0
        print(e)
        i=1
        for images, labels in train_load:
            if gpu==True:
                images=images.cuda()
                labels=labels.cuda()
            #= images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = vgg16.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            print("Still running",round(64*i/6552,2)*100,"%")
            i+=1
            running_loss += loss.item()
            #if i==10:
            #    break
        valid_loss=0
        acc=0
        vgg16.eval()
        i=1
        with torch.no_grad():
            for images, labels in valid_load:
                if gpu==True:
                    images=images.cuda()
                    labels=labels.cuda()
                #images, labels = images.to(device), labels.to(device)
                output=vgg16.forward(images)
                loss=criterion(output,labels)
                valid_loss+=loss.item()
                ps= torch.exp(output)
                top_p, top_class = ps.topk(1, dim = 1)
                equals = top_class == labels.view(*top_class.shape)
                acc+= torch.mean(equals.type(torch.FloatTensor)).item()
                #if i==10:
                #    break
        vgg16.train()
        print(f"Training loss: {running_loss/len(train_load)}")
        print(f"Accuracy: {acc / len(valid_load)}")
    save_checkpt(vgg16,IN_SIZE,OUT_SIZE,hidden_size,LR,BATCH_SIZE,epochs,optimizer,net,save_chkpt)

if __name__ == "__main__":
    data_dir,hidden_size,LR,epochs,gpu,net,save_chkpt = parse_args()
    train(data_dir,hidden_size,LR,epochs,gpu,net,save_chkpt)
