import torch;
import torch.nn as nn;
import datacollection as dc;

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

    x = torch.from_numpy(dc.collectTrainingX());
    y = torch.from_numpy(dc.collectTrainingY());

    model = TwoLayerNet(dc.TRAINX_SINGLE_DATA_SIZE, 50, 1);

    print('training on dataset');
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for z in range(500):

        y_res = model(x);

        loss = criterion(y_res, y);
        if z % 100 == 99:
            print(z, loss.item())

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

    print('training on dataset complete');
    
    print('testing on same dataset');
    


if __name__ == "__main__":
    main();
    