import os
import cv2
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable


DATA_DIR = os.getcwd() + "/train/"

x, y = [], []

for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".png"]:
    x.append(cv2.resize(cv2.imread(DATA_DIR + path), (1600, 1200)))
    with open(DATA_DIR + path[:-4] + ".txt", "r") as s:
        label = s.read()
    y.append(label.split("\n"))
x, y = np.array(x), np.array(y)

print(x.shape, y.shape)

# ENCODING LABELS HERE :---


binary_one_hot = MultiLabelBinarizer(classes=['red blood cell', 'difficult', 'gametocyte', 'trophozoite', 'ring', 'schizont', 'leukocyte'])
y_ = binary_one_hot.fit_transform(y)



#RANDOMLY SPLITTING 85% OF DATASET TO TRAIN AND 15% OF DATASET INTO TEST

X_train, X_test, y_train, y_test = train_test_split(x, y_, test_size=0.15)

np.save("X_train.npy", X_train); np.save("y_train.npy", y_train)

np.save("X_test.npy", X_test); np.save("y_test.npy", y_test)



# Loading Data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# converting from numpy to torch.Tensor
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)

X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)

# set up

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

##
X_train = X_train.view(len(X_train),3,1200,1600).float().to(device)
X_train.requires_grad = True
y_train = y_train.to(device)
X_test = X_test.view(len(X_test),3,1200,1600).float().to(device)
y_test = y_test.to(device)

# preparing for Data Loader

train_temp = []
test_temp = []

# for Train Loader
for count in range(len(X_train)):
    img = X_train[count]
    label = y_train[count]
    train_temp.append((img,label))
print(len(train_temp))


train_loader = DataLoader(dataset=train_temp,batch_size=25,shuffle=True)

# for Test Loader
for count in range(len(X_test)):
    img = X_test[count]
    label = y_test[count]
    test_temp.append((img,label))
print(len(test_temp))

test_loader = DataLoader(dataset=test_temp,batch_size=25,shuffle=False)


# CNN

num_epoc = 25
batch_size = 32
dropout = 0.5

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
        self.drop = nn.Dropout(dropout)
        self.linear2 = nn.Linear(400, 7)
        self.act1 = torch.relu
        self.act2 = torch.sigmoid

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act1(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act1(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act1(self.linear1(x.view(len(x), -1)))))
        return self.act2(self.linear2(x))


cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters())
criterion = nn.BCEWithLogitsLoss()

# Training

print("Training loop...")
for epoch in range(num_epoc):
    for i,(x_train, y_train) in enumerate(train_loader):

        x_train, y_train = Variable(x_train), Variable(y_train)

        optimizer.zero_grad()
        outputs = cnn(x_train)


        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (i+1)%32 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (i + 1, epoch, i + 1, len(X_train) // batch_size, loss.item()))

torch.save(cnn.state_dict(), "model_sauravmainali.pt")


# testing
print("Testing loop...")

cnn.load_state_dict(torch.load("model_sauravmainali.pt"))
cnn.eval()
with torch.no_grad():
    for i, (x_test, y_test) in enumerate(test_loader):
        x_test, y_test = Variable(x_test), Variable(y_test)
        prediction = cnn(x_test)
        loss = criterion(prediction, y_test)
        print(i+1, "===>", loss.item())


