# file to test import of the model

from __future__ import print_function, division

import torch.nn as nn
from torch.utils import data
from torchvision import datasets, models, transforms
import time, os, sys, copy, torch, torchvision
import numpy as np;
from PIL import Image

#set path to project
sys.path.append(os.path.realpath('.'));
import SW.datacollection as dc;

print('begin loading');
path = '../datasets';
# get total number of training labels
num_labels = sum(os.path.isdir(os.path.join(path, i)) for i in os.listdir(path));
print('num labels: ', num_labels);
#set device cpu/gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
print(device);

#perform data transforms
data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

#pre load resnet18 model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, num_labels)
model = model.to(device)

#load the model's weights
trained_model = torch.load('./ML/trained_model.pth.tar', map_location=device)
#import weights to model
model.load_state_dict(trained_model['state_dict'])
#switch model to eval mode as opposed to training mode
model.eval();
print('done loading')

#test the import in a loop
response = input('start testing? (y/n): ');

while (response == 'y'):
    #obtain live data
    test_data = dc.collectTestingX()[0];
    #get colormapped img
    depth_colormap_image = Image.fromarray(test_data);
    #perform data transforms
    preformatted = data_transforms(depth_colormap_image);

    #get label output
    ml_input = preformatted.unsqueeze(0).to(device);
    output = model(ml_input);
    prediction = int(torch.max(output.data, 1)[1]);

    #output to user
    print("predicted the letter: ", str(chr(prediction+97)));
    #softmax layer used for classifcation confidence values
    sm = nn.Softmax(dim=1);
    probabilities = sm(output);
    print('predicted actual: ', probabilities[0][prediction].item());
    print('predicited rounded probability: ', float("{0:.2f}".format(probabilities[0][prediction].item())))


    response = input("keep testing?: ");

print('done testing');
