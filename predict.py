import numpy as np
import time
import pandas as pd
import os, random
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision
import torchvision.models as models
torch.utils.model_zoo
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import torch.optim as optim
from collections import OrderedDict
import torch.nn.functional as F
import json
from PIL import Image
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('-ch','--checkpoint',action='store',help='Filepath of the previously made checkpoint',default='checkpoint.pth')
parser.add_argument('-g','--gpu',action='store_true',help='GPU used for prediction',default=True)
parser.add_argument('-f','--filepath',dest='filepath',help='Filepath of the image to be classified',default='/home/workspace/ImageClassifier/flowers/test/5/image_05169.jpg')
parser.add_argument('-t','--top_k',metavar='',help='Number of flower species with top probabilities',default=5,type=int)
parser.add_argument('-c','--cname_json',metavar='',help='Filepath of the json file to be used for category mapping of the flower species', default='cat_to_name.json')
args=parser.parse_args()

with open(args.cname_json,'r') as f:
    cat_to_name=json.load(f)
    
def load_checkpoint(file_path):
    checkpoint=torch.load(file_path)
    model=getattr(torchvision.models,checkpoint['Arch'])(pretrained=True)
    
    #Freeze Parameters
    for param in model.parameters():
        param.requires_grad = False
        
    classifier=nn.Sequential(OrderedDict([
                        ('fc1',nn.Linear(checkpoint['Input_Size'],checkpoint['Hidden_Layers'])),
                        ('relu1',nn.ReLU()),
                        ('drop',nn.Dropout(checkpoint['Drop'])),
                        ('fc2',nn.Linear(checkpoint['Hidden_Layers'],checkpoint['Output_Size'])),
                        ('output', nn.LogSoftmax(dim=1))
                        ]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.classifier.epochs = checkpoint['Epochs']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

#Image Preprocessing
def process_image(image):
    w, h = image.size
    image = image.resize((255, int(255*(h/w))) if w < h else (int(255*(w/h)), 255))
    
    w, h = image.size
    
    left = (w - 224)/2
    top = (h - 224)/2
    right = (w + 224)/2
    bottom = (h + 224)/2
    image = image.crop((left, top, right, bottom))
    
    image = np.array(image)
    
    image = image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image=(image-mean)/std
    
    image = image.transpose((2, 0, 1))
    return image

def predict(path, model, topk=7):
    model.eval()
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()
    else:
        model = model.cpu()
        
    im = Image.open(path)
    array = process_image(im)
    tensor = torch.from_numpy(array)
    
    if cuda:
        conv = Variable(tensor.float().cuda())
    else:       
        conv = Variable(tensor)
    
    conv = conv.unsqueeze(0)
    conv_f=model.forward(conv)
    ps=torch.exp(conv_f).data.topk(topk)
    prob=ps[0].cpu()
    classes=ps[1].cpu()   
    index_to_class={model.class_to_idx[key]: key for key in model.class_to_idx}
    m_idx_c = list()
    
    for ll in classes.numpy()[0]:
        m_idx_c.append(index_to_class[ll])     
    return prob.numpy()[0], m_idx_c

model=load_checkpoint(args.checkpoint)
gpu=args.gpu

image_path=args.filepath
prob,classes=predict(image_path,model,args.top_k)
print('Probabilities: ',prob)
print('Flower Species: ',[cat_to_name[x] for x in classes])
print('Species Index: ',classes)

#Saving the classification plot as predict.png
max_idx = np.argmax(prob)
max_prob = prob[max_idx]
label = classes[max_idx]
print('\033[1m' + '\nFlower Species With Highest Probability: ',cat_to_name[label])
print('\033[0m')
#Defining the probability plot
f_1 = plt.figure(figsize=(7,7))
axis_1 = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)
axis_2 = plt.subplot2grid((15,9), (9,2), colspan=5, rowspan=5)
axis_1.axis('off')
ll = []
for class_idx in classes:
    ll.append(cat_to_name[class_idx])
y_pos = np.arange(5)
plt.barh(y_pos, prob, xerr=0, align='center', color='green')
plt.title('Prediction Plot')
plt.yticks(y_pos,ll)
plt.xlabel('Probabilities')
plt.ylabel('Predicted flower species')
axis_2.invert_yaxis()
plt.savefig('/home/workspace/prediction.png')