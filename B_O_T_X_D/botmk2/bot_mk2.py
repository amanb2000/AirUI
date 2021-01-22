from librosa.util.utils import frame
from .ml import snufflupugus, CNN, CLASSES
from .frame_generator import frame_visualizer, generate_spectrogram, segment_spectrogram
import cv2
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import torch

DEBUG = False
DEBUG_PATH = '133.png'

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def main():
    # Just a little debug condition at the top :)
    if DEBUG:
        test = cv2.imread(DEBUG_PATH, 1)
        frame_visualizer(test, snufflupugus(test), CLASSES, save=True, fignum=100001)
        return
    
    mean = [0.4275, 0.4275, 0.4275]
    std = [0.2293, 0.2293, 0.2293]
    trans = transforms.Compose([transforms.CenterCrop((100, 65)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    
    dataset = torchvision.datasets.ImageFolder(\
            root='data', \
            transform=trans)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    # Lets start with making the spectrogram from the audio
    # spec = generate_spectrogram('long_spec_wav.wav')
    # Now let us segment that bad boi ;)
    # segments = segment_spectrogram(spec)
    # Finally we iterate, predict and plot
    i = 0
    imgs, labels = iter(loader).next()
    # for seg in segments:
    for img in imgs:      
        res = snufflupugus(img)
        img[0] = img[0] * std[0] + mean[0]     # De-normalize
        img[1] = img[1] * std[1] + mean[1]     # De-normalize
        img[2] = img[2] * std[2] + mean[2]     # De-normalize
        npimg = img.numpy()
        if res[0][0] < res[0][3]:
            frame_visualizer(np.transpose(npimg, [1,2,0]), res, CLASSES, save=True, fignum=i)
        i = i + 1
    return

if __name__ == '__main__':
    main()