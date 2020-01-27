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
y = dc.collectTrainingY();
num_labels = len(set(y));

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
print(device);

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, num_labels)
model = model.to(device)

trained_model = torch.load('trained_model.pth.tar', map_location=device)
model.load_state_dict(trained_model['state_dict'])

model.eval();
print('done loading')

response = input('start testing? (y/n): ');

while (response == 'y'):
    test_data = dc.collectTestingX();
    three_channel_data = [];
    three_channel_data.append([test_data, test_data, test_data]);
    ml_input = torch.Tensor(three_channel_data).to(device);

    output = model(ml_input);

    prediction = int(torch.max(output.data, 1)[1]);
    print("predicted the letter: ", str(chr(prediction+97)));

    response = input("keep testing?: ");

print('done testing');
