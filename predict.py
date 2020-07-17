# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 20:09:10 2020

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
import PIL
import numpy as np

DFLT_IM_PATH="/home/workspace/ImageClassifier/flowers/train/1/image_06734.jpg"
DFLT_CHKPT_PATH="/home/workspace/ImageClassifier/checkpoint.pth"
DFLT_TOPK=5
DFLT_CAT_NAMES='cat_to_name.json'
DFLT_GPU=True
def parse_args():
    parser = argparse.ArgumentParser(description = "Training")
    parser.add_argument("--im_path",type=str,default=DFLT_IM_PATH,help="Path")
    parser.add_argument("--chkpt_path",type=str,default=DFLT_CHKPT_PATH,help="Path")
    parser.add_argument("--topk",type=int,default=DFLT_TOPK,help="Path")
    parser.add_argument("--cat_names",type=str,default=DFLT_CAT_NAMES,help="Path")
    parser.add_argument("--gpu",type=bool,default=DFLT_GPU,help="Path")
    args=parser.parse_args()
    return args.im_path,args.chkpt_path,args.topk,args.cat_names,args.gpu

def load_chkpt(chkpt_path):
    checkpoint = torch.load(chkpt_path)
    model = getattr(models, checkpoint['net'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def get_cat_names(cat_names):
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)
    return(cat_to_name)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    i_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(244), 
                                     transforms.ToTensor(),])
    np_image = np.array(i_transform(image).float())

    np_image=(np.transpose(np_image,(1,2,0))-np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])
    np_image = np.transpose(np_image, (2, 0, 1))
    return(np_image)

def predict_topk(im_path, model, topk,gpu,cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    if gpu==True:
        model.cuda()
    model.eval()
    image = PIL.Image.open(im_path)
    np_image=process_image(image)
    tensor_image=torch.from_numpy(np_image).unsqueeze_(0).float()
    if gpu==True:
        tensor_image=tensor_image.cuda()
    logp = model.forward(tensor_image)
    prob = torch.exp(logp)    
    top_p, top_classes = prob.topk(topk, dim = 1)
    idx = {model.class_to_idx[c]: c for c in model.class_to_idx}
    top_c=[cat_to_name[idx[l]] for l in  top_classes.cpu().detach().numpy()[0]]
    print(top_p.cpu().detach().numpy()[0],top_c)
    return(top_p.cpu().detach().numpy()[0],top_c)

def finalpred(im_path, chkpt_path, top_k, cat_names, gpu):
    model=load_chkpt(chkpt_path)
    cat_to_name=get_cat_names(cat_names)
    top_p,top_c=predict_topk(im_path, model, top_k,gpu,cat_to_name)

if __name__ == "__main__":
    im_path, chkpt_path, top_k, cat_names, gpu = parse_args()
    
    finalpred(im_path, chkpt_path, top_k, cat_names, gpu)