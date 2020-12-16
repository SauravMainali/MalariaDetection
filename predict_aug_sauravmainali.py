import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable


class MyDataset(Dataset):

    def __init__(self, list_dir_x, img_width, img_height, transform=None):

        self.list_dir_x = list_dir_x
        self.img_width = img_width
        self.img_height = img_height
        self.transform = transform

        self.data = list_dir_x  # list of image_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path)
        img = img.resize((self.img_width, self.img_height))

        #Transform if not None
        if self.transform is not None:
            img = self.transform(img)

        return img # returns resized image





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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    images_path = np.array(images_path)

    image_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(1),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor()])



    test_dataset = MyDataset(images_path, 1600, 1200, image_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    cnn = CNN()
    cnn.load_state_dict(torch.load("model_sauravmainali.pt"))
    cnn.eval()

    outs =[]

    for batch_data in test_loader:
        batch_data = Variable(batch_data)
        prediction = cnn(batch_data)
        outs.append(prediction.cpu())

    outs = torch.cat(outs)
    return outs

#predict(["train/cells_0.png","train/cells_1.png"])

