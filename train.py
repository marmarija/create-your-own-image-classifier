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
    parser = argparse.ArgumentParser (description = 'Train.py')

    parser.add_argument ('data_dir', help = 'Add Data Directory', default='flowers', type = str)
    parser.add_argument ('--save_dir', help = 'Provide saving directory. Optional argument', default = '', type = str)
    parser.add_argument ('--arch', dest = 'arch', help = 'We can use VGG16', default='vgg16', type = str)
    parser.add_argument ('--learn_rate', help = 'Learning rate', default=0.001, type = float)
    parser.add_argument ('--hidden_layer', help = 'Hidden Layers', default=4098, type = int)
    parser.add_argument ('--epochs', help = 'Epoches number', default=3, type = int)
    parser.add_argument ('--GPU', help = "Use GPU", type = str)


    args = parser.parse_args ()
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = {
        'training': transforms.Compose([
            transforms.RandomRotation(60),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ]),
        'testing': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
    }
    image_datasets = {
        'training_data': datasets.ImageFolder(train_dir, transform = data_transforms['training']),
        'validation_data': datasets.ImageFolder(valid_dir, transform = data_transforms['validation']),
        'test_data': datasets.ImageFolder(valid_dir, transform=data_transforms['testing'])
    }
    dataloaders = {
        'training_loader': torch.utils.data.DataLoader(image_datasets['training_data'], batch_size = 64, shuffle = True ),
        'validation_loader': torch.utils.data.DataLoader(image_datasets['validation_data'], batch_size = 32, shuffle = True ),
        'test_loader': torch.utils.data.DataLoader(image_datasets['test_data'], batch_size = 32, shuffle = True )
    }

                    
    arch = args.arch
    learn_rate = args.learn_rate
    hidden_layer_1 = args.hidden_layer
    def neural_network_set(structure=arch, dropoput=0.5, hidden_layer = hidden_layer_1, lr = learn_rate):
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(40960,hidden_layer)),
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

    model, optimizer, criterion = neural_network_set(arch)
    if args.GPU == 'GPU':
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)

    def neural_network_training(model, trainloader, epochs, print_every, criterion, optimizer, device = device):
        step = 0
        for e in range(epochs):
            running_loss = 0
            print(e)
            for image, label in trainloader:
                print('inside for')
                step += 1
                image,label = image.to(device),label.to(device)
                optimizer.zero_grad()
                output = model.forward(image)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if step % print_every == 0:
                    model.eval()
                    vlost = 0
                    vaccuracy=0
                    print('step/every')
                    for image_1, label_1 in dataloaders['validation_loader']:
                        optimizer.zero_grad()
                        image_1,label_1 = image_1.to(device),label_1.to(device)
                        with torch.no_grad():
                            output = model.forward(image_1)
                            vlost = criterion(output,label_1)
                            ps = torch.exp(output).data
                            equality = (label_1.data == ps.max(1)[1])
                            vaccuracy += equality.type_as(torch.FloatTensor()).mean()
                        
                    vlost = vlost / len(dataloaders['validation_loader'])
                    vaccuracy = vaccuracy / len(dataloaders['validation_loader'])
                    
                    print(
                        "Epoch: {}/{}... ".format(e+1, epochs),
                        "Loss: {:.4f}".format(running_loss/print_every),
                        "Validation Lost {:.4f}".format(vlost),
                        "Accuracy: {:.4f}".format(vaccuracy)
                    )
                    running_loss = 0

    epochs = args.epochs
    learn_rate = args.learn_rate
    save_dir = args.save_dir
    neural_network_training(model, dataloaders['training_loader'], epochs, 40, criterion, optimizer, device)  

    def check_test_accuracy(model, data, cuda=False):
        model.eval()
        model.to(device='cuda') 
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in (dataloaders[data]):
                image, label = data
                image, label = image.to(device), label.to(device)
                output = model(image)
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        
        accuracy = 100 * correct / total
        print('Network accuracy on the test images: %d %%' % accuracy)
        
    check_test_accuracy(model, 'test_loader', True)

    model.class_to_idx = image_datasets['training_data'].class_to_idx
    model.cpu
    torch.save({
        'structure': arch,
        'hidden_layer': hidden_layer_1,
        'learning_rate': learn_rate,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
        },'./'+save_dir+'checkpoint.pth')

if __name__ == '__main__':
    main()
