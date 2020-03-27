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
from PIL import Image


print('begin loading');
path = 'datasets';
num_labels = sum(os.path.isdir(os.path.join(path, i)) for i in os.listdir(path));
print('num labels: ', num_labels);
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
print(device);

data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

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
    depth_colormap_image = Image.fromarray(test_data);
    preformatted = data_transforms(depth_colormap_image);
    # look at how the transferLearning script modifies the input and match that for this case

    ml_input = preformatted.unsqueeze(0);


    output = model(ml_input);
    prediction = int(torch.max(output.data, 1)[1]);
    print("predicted the letter: ", str(chr(prediction+97)));
    #sm = nn.Softmax();
    #probabilities = sm(output);
    #print('probabilities: ', probabilities);
    #print('predicited probability: ', probabilities[0][prediction])

    response = input("keep testing?: ");

print('done testing');
