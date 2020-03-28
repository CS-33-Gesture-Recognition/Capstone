import torch;
import torch.nn as nn;
import datacollection as dc;
import numpy as np;

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
print('running on device: ' + str(device));


class TwoLayerNet(nn.Module):
    def __init__(self, data_in, H, data_out):
        super(TwoLayerNet, self).__init__();
        self.linear1 = nn.Linear(dc.TRAINX_SINGLE_DATA_SIZE, H);
        self.linear2 = nn.Linear(H, 1);

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0);
        y_pred = self.linear2(h_relu);
        return y_pred;

def main():

    dc.gatherCameraImage();


if __name__ == "__main__":
    main();
