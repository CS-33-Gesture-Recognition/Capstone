from __future__ import print_function, division

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import datacollection as dc;
import numpy as np;
import torch
from torch.utils import data


def LabelConverter(s):
    if s == 'a':
        return 0
    else:
        return 1
    return -1;

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");

# need an intermediate step where all the data is shuffled and then from there we split train/test/val
# consider randomizing, and then k-fold

x = dc.collectTrainingX();
y = dc.collectTrainingY();

reshaped_x = [];
relabeled_y = [];

#relabeling the classification vector
for i in y:
    relabeled_y.append(LabelConverter(i));

#reshaping the data itself to height x width dims
for i in x:
    temp = np.reshape(i, (480, 640));
    reshaped_x.append(temp);

#split train data
train_data = [];
train_labels = []

for i in range(0,20,1):
# this gets me the single channel, now I want 3
#   train_data.append([reshaped_x[i]]);
# i think this gets me 3, and now the script is working
   train_data.append([reshaped_x[i], reshaped_x[i], reshaped_x[i]]);
   train_labels.append(relabeled_y[i]);

for i in range(30,50,1):
   train_data.append([reshaped_x[i], reshaped_x[i], reshaped_x[i]]);
   train_labels.append(relabeled_y[i]);

#create data set
tensor_train_x = torch.Tensor(train_data); # transform to torch tensor
tensor_train_y = torch.Tensor(train_labels);
train_dataset = data.TensorDataset(tensor_train_x, tensor_train_y);

#create training loader
train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=4);

#split val data
val_data = [];
val_labels = [];

for i in range(20,30,1):
   val_data.append([reshaped_x[i], reshaped_x[i], reshaped_x[i]]);
   val_labels.append(relabeled_y[i]);

for i in range(50,60,1):
   val_data.append([reshaped_x[i], reshaped_x[i], reshaped_x[i]]);
   val_labels.append(relabeled_y[i]);

#create dataset
tensor_val_x = torch.Tensor(val_data); # transform to torch tensor
tensor_val_y = torch.Tensor(val_labels);
val_dataset = data.TensorDataset(tensor_val_x, tensor_val_y);

#create val loader
val_loader = data.DataLoader(val_dataset, shuffle=True, batch_size=4);

datasets = {};
datasets['train'] = train_dataset;
datasets['val'] = val_dataset;

#create pair of loaders
dataloaders = {};
dataloaders['train'] = train_loader;
dataloaders['val'] = val_loader;

#get sizes
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).long();

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

# need a test function - from dibz

#testing metrics
"""
ROC-AUC curves
PR
Confusion Matrix
    Sensitivity - ability to correctly predict
    Specificity -
    Informedness - combined specificity & sensitivity into one metric

"""


# need to know how to export and re-use
# use torch.save to export the model

# maybe try to get this working with live data before moving forward and collecting all data

# torch.save(mode_ft);

print("done");
