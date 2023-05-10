import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import numpy as np
import json
import PIL
from PIL import Image

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from collections import OrderedDict

import argparse

import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='store', help='Name of trained model', default='checkpoint.sh', type=str)
    parser.add_argument('--json', action='store',help='JSON file with names of the classes.', default='', type=str)
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available', default='gpu', type='str')
    parser.add_argument('--image_path',action='store', help='Image Path', default='flowers/test/1/image_06752.jpg' type=str)
    parser.add_argument('--topk', action='store',type=int, help='Select number of classes you wish to see in descending order.', default=5, type=int)

    args = parser.parse_args()
    json_file = args.json
    image_path = args.image_path
    topk = args.topk
    gpu = args.gpu
    checkpoint = args.checkpoint

    def neural_network_set(structur, dropoput, hidden_layer, lr):
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(25088,hidden_layer)),
                                ('relu1', nn.ReLU()),
                                ('drop1',nn.Dropout(dropoput)),
                                ('fc2', nn.Linear(hidden_layer, 1024)),
                                ('relu2', nn.ReLU()),
                                ('drop2',nn.Dropout(dropoput)),
                                ('fc3', nn.Linear(1024, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr)
        
        return model, optimizer, criterion

    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
        print(cat_to_name)

    def load_checkpoint(path):
        checkpoint = torch.load(path)
        structure = checkpoint['structure']
        hidden_layer = checkpoint['hidden_layer']
        learning_rate = checkpoint['learning_rate']
        model,_,_ = neural_network_set(structure , 0.5, hidden_layer, learning_rate)
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
        
        return model
 
    model = load_checkpoint(checkpoint)
    print(model)     

    def process_image(image):
        open_image = Image.open(image)
        
        transform_image = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
        pil_image = transform_image(open_image)
    
        return pil_image

    test_image = image_path
    test_image = process_image(test_image)
    print(test_image)

    def imshow(image, ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots()
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        
        return ax
    imshow(process_image(image_path))
    def predict(image_path, model, topk):
        torch_image = process_image(image_path)
        torch_image = torch_image.unsqueeze_(0)
        torch_image = torch_image.float()
        with torch.no_grad():
            output = model.forward(torch_image)
        
        probability = F.softmax(output.data, dim = 1)
        return probability.topk(topk)

    probs, classes = predict(image_path,model, topk)
    print(probs)
    print(classes)
    def sanity_checker(image_path):
        i = 1
        prediction_prob = predict(image_path,model)
        image = process_image(path)
        
        plt.rc('figure', figsize=(8,8))
        plt.subplot(211)
        sc_img = imshow(image, ax=plt)
        sc_img.axis('off')
        sc_img.title(cat_to_name[str(i)],y=-0.1)
        sc_img.show()
        
        a = np.array(prediction_prob[0][0])
        b = [cat_to_name[str(index+1)] for index in np.array(prediction_prob[1][0])]
        
        
        N=float(len(b))
        fig,ax = plt.subplots(figsize=(8,3))
        width = 0.8
        tickLocations = np.arange(N)
        ax.bar(tickLocations, a, width, linewidth=4.0, align = 'center')
        ax.set_xticks(ticks = tickLocations)
        ax.set_xticklabels(b)
        ax.set_xlim(min(tickLocations)-0.6,max(tickLocations)+0.6)
        ax.set_yticks([0.2,0.4,0.6,0.8,1,1.2])
        ax.set_ylim((0,1))
        ax.yaxis.grid(True)
        
        plt.show()
    sanity_checker(image_path)    