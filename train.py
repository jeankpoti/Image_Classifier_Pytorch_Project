import argparse
from collections import OrderedDict
import json

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from PIL import Image

import seaborn as sns
import time
from torch import nn
from torch import optim
import torch
from torch.autograd import Variable
import torch.utils.data as data

from torchvision import datasets, transforms, models
import torchvision

from tqdm import tqdm




# Initialize  argparse
parser = argparse.ArgumentParser (description = "Command line application with parser for train.py")

# Define positional and optional arguments
parser.add_argument ('--data_dir', help = 'Data directory', type = str)
parser.add_argument ('--save_dir', help = 'Directory to save', type = str)
parser.add_argument ('--arch', help = 'Use alexnet or vgg13', type = str)
parser.add_argument ('--lrn', help = 'Learning rate', type = float)
parser.add_argument ('--hidden_units', help = 'Hidden units', type = int)
parser.add_argument ('--epochs', help = 'Number of epochs', type = int)
parser.add_argument ('--GPU', help = "Choose GPU or CPU", type = str)

args = parser.parse_args ()

# Set directory
#data_dir = args.data_dir
#train_dir = data_dir + '/train'
#valid_dir = data_dir + '/valid'
#test_dir = data_dir + '/test'

# Set what device to use
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'
    
    
if args.data_dir:
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])


    # Load the datasets with ImageFolder
    train_set = datasets.ImageFolder(root = "{}/train".format(args.data_dir), transform=train_transforms)
    test_set = datasets.ImageFolder(root = "{}/test".format(args.data_dir), transform=test_transforms)
    valid_set = datasets.ImageFolder(root = "{}/valid".format(args.data_dir), transform=valid_transforms)
    image_datasets = [train_set, test_set, valid_set]
    trainloader = data.DataLoader(train_set, batch_size = 4, shuffle=True)
    testloader = data.DataLoader(test_set, batch_size = 4, shuffle=True)
    validloader = data.DataLoader(valid_set, batch_size = 4, shuffle=True)
    dataloaders = [train_loader, test_loader, valid_loader]

    
#Label mapping
#with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    

    # Load model
    def load_model (arch, hidden_units):
        if arch == 'vgg13':
            model = models.vgg13 (pretrained = True)
            for param in model.parameters():
                param.requires_grad = False
            if hidden_units:
                classifier = nn.Sequential  (OrderedDict ([
                                ('fc1', nn.Linear (25088, 4096)),
                                ('relu1', nn.ReLU ()),
                                ('dropout1', nn.Dropout (p = 0.3)),
                                ('fc2', nn.Linear (4096, hidden_units)),
                                ('relu2', nn.ReLU ()),
                                ('dropout2', nn.Dropout (p = 0.3)),
                                ('fc3', nn.Linear (hidden_units, 102)),
                                ('output', nn.LogSoftmax (dim =1))
                                ]))
            else:
                classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (25088, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, 2048)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (2048, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        else: 
            arch = 'alexnet' 
            model = models.alexnet (pretrained = True)
            for param in model.parameters():
                param.requires_grad = False
            if hidden_units: 
                classifier = nn.Sequential  (OrderedDict ([
                                ('fc1', nn.Linear (9216, 4096)),
                                ('relu1', nn.ReLU ()),
                                ('dropout1', nn.Dropout (p = 0.3)),
                                ('fc2', nn.Linear (4096, hidden_units)),
                                ('relu2', nn.ReLU ()),
                                ('dropout2', nn.Dropout (p = 0.3)),
                                ('fc3', nn.Linear (hidden_units, 102)),
                                ('output', nn.LogSoftmax (dim =1))
                                ]))
            else:
                classifier = nn.Sequential  (OrderedDict ([
                             ('fc1', nn.Linear (9216, 4096)),
                             ('relu1', nn.ReLU ()),
                             ('dropout1', nn.Dropout (p = 0.3)),
                             ('fc2', nn.Linear (4096, 2048)),
                             ('relu2', nn.ReLU ()),
                             ('dropout2', nn.Dropout (p = 0.3)),
                             ('fc3', nn.Linear (2048, 102)),
                             ('output', nn.LogSoftmax (dim =1))
                            ]))
        model.classifier = classifier
        return model, arch


    # Validation function for validation set
    def validation(model, validloader, criteron):
        valid_loss = 0
        accuracy = 0
        for inputs, labels in validloader:

            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            valid_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy = equality.type(torch.FloatTensor).mean()

        return valid_loss, accuracy 



    model, arch = load_model (args.arch, args.hidden_units)


    # Initialize criterion
    criterion = nn.NLLLoss()

    # Initialize optimizer
    if args.lrn:
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.lrn) # learning rate was given
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001) # learning rate was not given





    if args.epochs:
        epochs = args.epochs 
    else:
        epochs = 7

    steps = 0
    print_every = 40

    # Using available device
    model.to(device) 


    for e in range(epochs):
        running_loss = 0
        model.train()
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)
            model.to(device)

            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, testloader, criterion)

                print("Epoch: {}/{}..".format(e+1, epochs),
                      "Training Loss: {:.3f}..".format(running_loss/print_every),
                      "Validation Loos: {:.3f}..".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}..".format(accuracy/len(validloader)))

                running_loss = 0

                model.train()



    # Save the checkpoint 
    model.to('cpu')

    model.class_to_idx = train_set.class_to_idx

    checkpoint = {'input_size': 2048,
                  'output_size': 108,
                  'batch_size': 64,
                  'epochs': epochs,
                  'model': models.alexnet(pretrained=True),
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx
                 }

    if args.save_dir:   
        torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
    else:    
       torch.save(checkpoint, 'checkpoint.pth')

