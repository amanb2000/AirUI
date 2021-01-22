# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
import queue

CLASSES = ['Circle Scratch', 'Fingernail Tap', 'Fingertip Tap', 'Silence', 'Vertical Scratch', 'W Scratch']

# Model Init
class CNN(nn.Module):
  def __init__(self, input_dims, numOfKernels, numOfNeurons, kernelSize, numOfConvLayers, batchNorm):
    super(CNN, self).__init__()         
    self.numOfKernels = numOfKernels
    self.batchNorm = batchNorm
    self.numOfConvLayers = numOfConvLayers

    # Convolutional Layers
    self.conv1 = nn.Conv2d(3,numOfKernels, kernelSize)
    self.conv2 = nn.Conv2d(numOfKernels,numOfKernels, kernelSize)
    self.conv_BN = nn.BatchNorm2d(numOfKernels)

    # Determine the output size after the convolutional layer
    fullLayerSize_x = input_dims[1]
    fullLayerSize_y = input_dims[0]
    for i in range (self.numOfConvLayers):
      fullLayerSize_x = (fullLayerSize_x-kernelSize+1)//2
      fullLayerSize_y = (fullLayerSize_y-kernelSize+1)//2

    # Error check the output size
    if fullLayerSize_x <= 0 or fullLayerSize_y <= 0:
      raise Exception("Too many convolutional layer for the input size, please decrease numOfConvLayers.")

    # Fully connected layers
    self.fc1 = nn.Linear(numOfKernels*fullLayerSize_x*fullLayerSize_y, numOfNeurons)
    self.fc1_BN = nn.BatchNorm1d(numOfNeurons)
    self.pool = nn.MaxPool2d(2,2)
    self.fc2 = nn.Linear(numOfNeurons, 6)
    self.fc2_BN = nn.BatchNorm1d(6)

  def forward(self, x):
    activation = F.relu   

    if self.batchNorm == True:
      x = self.pool(activation(self.conv_BN(self.conv1(x))))
      for i in range (self.numOfConvLayers - 1):
        x = self.pool(activation(self.conv_BN(self.conv2(x))))
      x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
      x = activation(self.fc1_BN(self.fc1(x)))
      x = self.fc2_BN(self.fc2(x))
    else: 
      x = self.pool(activation(self.conv1(x)))
      for i in range (self.numOfConvLayers - 1):
        x = self.pool(activation(self.conv2(x)))
      x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
      x = activation(self.fc1(x))
      x = self.fc2(x)
    return x

from torch.utils.data import IterableDataset

class MyDataset(IterableDataset):
    def __init__(self, image_queue, transforms=None):
      self.queue = image_queue
      self.transforms = transforms

    def read_next_image(self):
        while self.queue.qsize() > 0:
            if self.transforms is not None:
              # yield self.transforms(self.queue.get())
              pass
            # you can add transform here
            yield self.queue.get()
        return None

    def __iter__(self):
        return self.read_next_image()

def snufflupugus(spec_in):
    # Get model
    model = CNN(input_dims=(100, 65), numOfKernels=26, numOfNeurons=100, kernelSize=3, numOfConvLayers=2, batchNorm=True)
    model.load_state_dict(torch.load('crop_best_dict_nosplit_f.pt'))
    model.eval()
    # trans = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop((100, 65)), transforms.ToTensor(), transforms.Normalize([0.4275, 0.4275, 0.4275], [0.2293, 0.2293, 0.2293])])
    # clean_scan = trans(spec_in)
    clean_scan = spec_in
    predict = model(clean_scan.reshape(1, 3, 100, 65))
    return predict