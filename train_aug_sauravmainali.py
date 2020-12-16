import os
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable


######

# Set Up

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#########

DATA_DIR = os.getcwd() + "/train/"

x, y = [], []
x = [f for f in os.listdir(DATA_DIR) if f[-4:] == ".png"] # x is 'list of image names'

for pic in x:
    with open(DATA_DIR + pic[:-4] + ".txt", "r") as s:
        label = s.read()
    y.append(label.split("\n")) # y is 'list of targets' of corresponding x

x, y = np.array(x), np.array(y)



# Encoding the targets in binary form

binary_one_hot = MultiLabelBinarizer(classes=['red blood cell', 'difficult', 'gametocyte', 'trophozoite', 'ring', 'schizont', 'leukocyte'])
y_ = binary_one_hot.fit_transform(y) # binary one hot encoded form of the target
y_ = np.array(y_,dtype=float)
########

# Splitting 85% of the data for training and rest for testing

X_train, X_test, y_train, y_test = train_test_split(x, y_, test_size=0.05)

########

# Preparing the CNN Network
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



########
# Inheriting Dataset and Overiding methods - __len__() and __getitem__()
class MyDataset(Dataset):

    def __init__(self, temp_x,temp_y, img_width, img_height, transform=None):

        self.temp_x = temp_x
        self.temp_y = temp_y
        self.img_width = img_width
        self.img_height = img_height
        self.transform = transform
        temp_data = list()
        for count in range(temp_x.shape[0]):
            temp_data.append((temp_x[count],temp_y[count]))

        self.data = temp_data  # list of (image_name, target) tuples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, img_label = self.data[index]
        img = Image.open(DATA_DIR + str(img_name))
        img = img.resize((self.img_width, self.img_height))

        #Transform if not None
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(img_label)
        return img, label # returns resized image and label

#########

# Data Augmentation using torchvision.transforms

image_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(1),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor()])

# image_transforms2 = transforms.Compose([
#     transforms.ToTensor()])

img_width = 1600
img_height = 1200
batch_size = 32
dropout = 0.5
num_epoc = 25


train_dataset = MyDataset(X_train,y_train,img_width,img_height,image_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


test_dataset = MyDataset(X_test,y_test,img_width,img_height,image_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#######

# Training

cnn = CNN().to(device)
optimizer = torch.optim.Adam(cnn.parameters())
criterion = nn.BCEWithLogitsLoss()

# print("Training loop...")


for epoch in range(num_epoc):
    for i, (x_train, y_train) in enumerate(train_loader):


        optimizer.zero_grad()
        print(i)
        outputs = cnn(x_train)

        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()


    print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
        % (i + 1, epoch, i + 1, len(X_train) // batch_size, loss.item()))

torch.save(cnn.state_dict(), "model_sauravmainali.pt")

print("Testing loop...")

cnn.load_state_dict(torch.load("model_sauravmainali.pt"))
cnn.eval()
with torch.no_grad():
    for i, (x_test, y_test) in enumerate(test_loader):

        prediction = cnn(x_test)
        loss = criterion(prediction, y_test)
        print(i+1, "===>", loss.item())
