import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import pandas as pd
import json
import argparse



# Initialize  argparse
parser = argparse.ArgumentParser (description = "Command line application with parser for predict.py")

# Define positional and optional arguments
parser.add_argument ('image_dir', help = 'Image path', type = str)
parser.add_argument ('checkpoint', help = 'checkpoint path', type = str)
parser.add_argument ('--topk', help = 'Top K classes', type = int)
parser.add_argument ('--category_names', help = 'JSON file that maps the class values to other category names', type = str)
parser.add_argument ('--GPU', help = 'Choose GPU or CPU', type = str)

args = parser.parse_args ()


#Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load (file_path)
    if checkpoint ['arch'] == 'alexnet':
        model = models.alexnet (pretrained = True)
    else: 
        model = models.vgg13 (pretrained = True)
        model.classifier = checkpoint ['classifier']
        model.load_state_dict (checkpoint ['state_dict'])
        model.class_to_idx = checkpoint ['mapping']

    for param in model.parameters():
        param.requires_grad = False 

    return model

# Function for processing image
def process_image(image):
    # TODO: Process a PIL image for use in a PyTorch model
    img_pil = Image.open(image)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(img_pil)
    
    return img_tensor
    
img = ('flowers/test/1/image_06743.jpg')
img = process_image(img)



model.class_to_idx = train_data.class_to_idx

ctx = model.class_to_idx

def predict(image_path, model, topk=5):   
    model.to('cuda:0')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    probability = F.softmax(output.data,dim=1)
    
     # Top probs
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().cpu().numpy().tolist()[0] 
    top_labs = top_labs.detach().cpu().numpy().tolist()[0]

    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers


# Image path
file_path = args.image_dir

# Device choosing option
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

#load JSON file if provided, else load default file name
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass
    

#loading model from checkpoint provided
model = loading_model (args.load_dir)


if args.topk:
    number_classe = args.topk
else:
    number_classe = 5

#calculate probabilities and classes
probs, classes = predict (file_path, model, nm_cl, device)


class_names = [cat_to_name [item] for item in classes]

for l in range (number_classe):
     print("Number: {}/{}.. ".format(l+1, number_classe),
            "Class name: {}.. ".format(class_names [l]),
            "Probability: {:.3f}..% ".format(probs [l]*100),
            )    