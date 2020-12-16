import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (5, 5), stride=5)
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(16, 32, (5, 5), stride=5)
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.linear1 = nn.Linear(32 * 12 * 16, 400)
        self.linear1_bn = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(0.5)
        self.linear2 = nn.Linear(400, 7)
        self.act1 = torch.relu
        self.act2 = torch.sigmoid

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act1(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act1(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act1(self.linear1(x.view(len(x), -1)))))
        return self.act2(self.linear2(x))




def predict(images_path):
    x = []
    for pic in images_path:
        x.append(cv2.resize(cv2.imread(pic), (1600, 1200)))
    x = np.array(x)
    x = torch.Tensor(x)
    x = x.view(len(x), 3, 1200, 1600).float()

    x_loader = DataLoader(dataset=x,batch_size=1,shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cnn = CNN()
    cnn.load_state_dict(torch.load("model_sauravmainali.pt"))
    cnn.eval()

    outs = []

    for batch_data in x_loader:

        batch_data = Variable(batch_data)
        prediction = cnn(batch_data)
        outs.append(prediction.cpu())

    outs =torch.cat(outs)
    return outs

#predict(["train/cells_0.png","train/cells_1.png"])


