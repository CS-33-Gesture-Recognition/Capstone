from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import shutil
import split_folders
from sklearn.metrics import confusion_matrix


#  normalization for testing
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#load in the testing data
data_dir = 'output_split'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}

class_names = image_datasets['test'].classes

#set the device (CPU or GPU) and obtain total number of labels
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_labels = len(class_names);

#obtain resnet 18 model to load in trained weight
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
#set the output size of the model
model.fc = nn.Linear(num_ftrs, num_labels)
model = model.to(device)
#load in the trained weights after training
trained_model = torch.load('trained_model.pth.tar', map_location=device)
model.load_state_dict(trained_model['state_dict'])
#set model to eval mode for classification
model.eval();
print('done loading')

# Initialize the prediction and label lists(tensors)
predlist=torch.zeros(0,dtype=torch.long, device=device)
lbllist=torch.zeros(0,dtype=torch.long, device=device)
for inputs, labels in dataloaders['test']:
    inputs = inputs.to(device)
    labels = labels.to(device).long();
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    predlist=torch.cat([predlist,preds.view(-1).to(device)])
    lbllist=torch.cat([lbllist,labels.view(-1).to(device)])

#compute and print confusion matrix
conf_mat=confusion_matrix(lbllist.cpu().numpy(), predlist.cpu().numpy())
print(conf_mat)

# Per-class accuracy print
class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
print(class_accuracy)
