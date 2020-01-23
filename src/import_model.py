from __future__ import print_function, division

import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import datacollection as dc;
import numpy as np;
import torch
from torch.utils import data
import datacollection as dc;


print('begin loading');

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 2)

trained_model = torch.load('trained_model.pth.tar', map_location=device)
model.load_state_dict(trained_model['state_dict'])

test_data = dc.collectTestingX();

three_channel_data = [];

three_channel_data.append([test_data, test_data, test_data]);

input = torch.Tensor(three_channel_data);

#print(input.size());

output = model(input);

print("raw data: ",output.data);

print("torch.max call: ", torch.max(output.data, 1)[1].numpy())

prediction = int(torch.max(output.data, 1)[1].numpy())

if (prediction == 0):
    print('predicted an a!')
elif (prediction == 1):
    print('predicted a b!')
else:
    print('something else happened, ask Dibz')
print('done loading')
