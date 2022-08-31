import numpy as np
import matplotlib.pyplot as plt
import os
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import argparse
torch.utils.model_zoo
from collections import OrderedDict
from PIL import Image
import time

#Using argparse variable
parser = argparse.ArgumentParser()
parser.add_argument("-d","--dataset",metavar='',help="Type the path of the dataset on which training has to be done",default="/home/workspace/ImageClassifier/flowers")
parser.add_argument("-a","--arch",metavar='',help="Choose which architecture to use.Type \"-a help\" to see the list of architectures",default='vgg16')
parser.add_argument("-e","--epochs",metavar='',help="Type the no of epochs(>=1)",default=2,type=int)
parser.add_argument("-D","--device",metavar='',help="Choose the device(CPU or GPU)",default="gpu")
parser.add_argument("-u","--hidden_layer",metavar='',help="Type the number of hidden layer(>=1)",default=1500,type=int)
parser.add_argument("-r","--learning_rate",metavar='',help="Type the learning rate(>0 && <1)",default=0.0001,type=float)                    
args = parser.parse_args()

architecture_list = ["vgg16","resnet18","alexnet"]
if(args.arch=='help'):
    print("The list of available architectures is as follows:")
    print("1) vgg16(Default architecture)")
    print("2) resnet18")
    print("3) alexnet")
    quit()
if(args.arch not in architecture_list):
    print("Invalid Architecture choosen!! Type \"python train.py -a help\" to see the list of architectures available")
    quit()
if(args.device=='gpu'):
    args.device='cuda'
else:
    args.device='cpu'

#Load the data   
data_dir = args.dataset
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

#Validation Data
data_transforms_validation = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

#Testing data
data_transforms_testing = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]) 

# TODO: Load the datasets with ImageFolder
#Training data
image_datasets_train = datasets.ImageFolder(train_dir,transform=train_transforms)
#Validation data
image_datasets_validation = datasets.ImageFolder(valid_dir,transform=data_transforms_validation)
#Testing data
image_datasets_testing = datasets.ImageFolder(test_dir,transform=data_transforms_testing) 

# TODO: Using the image datasets and the trainforms, define the dataloaders
#Training data
dataloaders_train = torch.utils.data.DataLoader(image_datasets_train, batch_size=32, shuffle=True)
#Validation data
dataloaders_validation = torch.utils.data.DataLoader(image_datasets_validation, batch_size=32)
#Testing data
dataloaders_testing = torch.utils.data.DataLoader(image_datasets_testing, batch_size=32)
                                             
#label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
                                             
#Building and training classifier                                             
#Choosing between different architectures                                             
if args.arch=='vgg16':
    model=models.vgg16(pretrained=True)
    input_size=25088                                         
elif args.arch=='resnet18':                                             
    model=models.resnet18(pretrained=True)
    input_size=512                                         
else:
    model=models.alexnet(pretrained=True)
    input_size=9216
                                             
#Freeze parameters
for param in model.parameters():
    param.requires_grad = False
                                             
#Hyperparameters
output_size=102
drop=0.2
steps=0
print_every=4
epochs=args.epochs
                                             
#Defining the architecture of classifier Neural network 
classifier=nn.Sequential(OrderedDict([
                        ('fc1',nn.Linear(input_size,args.hidden_layer)),
                        ('relu1',nn.ReLU()),
                        ('drop',nn.Dropout(drop)),
                        ('fc2',nn.Linear(args.hidden_layer,102)),
                        ('output', nn.LogSoftmax(dim=1))
                        ]))
                                             
#Replacing the corresponding inbuilt classifier with our own coded custom classifier 
model.classifier = classifier
                                             
criterion=nn.NLLLoss()
optimizer=optim.Adam(model.classifier.parameters(), lr=args.learning_rate)         
#Shifting the training process to either cpu or gpu
model.to(args.device)

for i in range(epochs):
    running_loss=0
    for images,labels in dataloaders_train:
        steps +=1
        images, labels = images.to(args.device), labels.to(args.device)#Shifting the images and labels to either cpu or gpu
        optimizer.zero_grad()
        output=model.forward(images)
        loss=criterion(output,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        
        if steps % print_every == 0:
            print("Epoch = {}/{}    ".format(i+1, epochs),
                  "Training Loss = {:.4f}".format(running_loss/print_every))    
        running_loss = 0
        
        #Calculating the accuracy
        if steps % 20*print_every == 0:
            corr = 0
            tot = 0
            for ii, (images, labels) in enumerate(dataloaders_validation):
                with torch.no_grad():
                    images, labels = images.to(args.device), labels.to(args.device)
                    
                # Forward and backward passes
                    output = model.forward(images)
                    _, predicted = torch.max(output.data, 1)
                    corr += (predicted == labels).sum().item()
                    tot += labels.size(0)
                    
            print("Accuracy: {0:.2f}%".format(corr/tot*100))

accuracy=0
t=0
corr=0
print("Testing trained network on test data")
for ii, (images, labels) in enumerate(dataloaders_testing):
    with torch.no_grad():
        images,labels=images.to(args.device),labels.to(args.device)
        output=model.forward(images)
        _,pred=torch.max(output.data,1)
        corr+=(pred==labels).sum().item()
        ps=torch.exp(output)
        equality=(labels.data==ps.max(dim=1)[1])
        accuracy+=equality.type(torch.FloatTensor).mean()
        t+=labels.size(0)
print("Total Images=",t)
print("Correctly Classified Images=",corr)
print("Accuracy={0:.2f}%".format(corr/t*100))

#Save Checkpoint
model.class_to_idx = image_datasets_train.class_to_idx

checkpoint={'Input_Size':input_size,
              'Output_Size':output_size,
              'Hidden_Layers':args.hidden_layer,
              'Drop':drop,
              'Epochs':epochs,
              'Arch':args.arch,
              'Loss':loss,
              'model_state_dict':model.state_dict(),
              'optimizer_state_dict':optimizer.state_dict(),
              'class_to_idx':model.class_to_idx}
torch.save(checkpoint,'checkpoint.pth')