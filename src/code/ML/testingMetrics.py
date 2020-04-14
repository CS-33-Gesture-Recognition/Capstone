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



# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '../output_split'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['test']}


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}

class_names = image_datasets['test'].classes

print("class_names: ", class_names);
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('num classes: ', len(class_names));
num_labels = len(class_names);

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, num_labels)
model = model.to(device)

trained_model = torch.load('./ML/trained_model.pth.tar', map_location=device)
model.load_state_dict(trained_model['state_dict'])

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
    #print('label: ', labels.data.numpy(), '    preds: ',preds.numpy());
    #confusion_matrix()
    predlist=torch.cat([predlist,preds.view(-1).to(device)])
    lbllist=torch.cat([lbllist,labels.view(-1).to(device)])

#print('label list: ', lbllist, '   pred list: ', predlist);

conf_mat=confusion_matrix(lbllist.cpu().numpy(), predlist.cpu().numpy())
print(conf_mat)

# Per-class accuracy
class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
print(class_accuracy)
