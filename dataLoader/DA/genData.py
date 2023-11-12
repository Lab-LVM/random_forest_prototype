import torch
import torchvision
import os
import torch.utils.data
import pickle
import glob
import numpy as np
from PIL import Image

class office31(torch.utils.data.Dataset):

    def __init__(self, rootFolder):
        self.rootFolder = os.path.join(os.path.join(rootFolder, 'data'), 'DA')
        self.datafile = os.path.join(self.rootFolder, 'office31.pickle')

        self.images = []
        self.labels = []
        self.domains = []
        self.files = []

        if os.path.isfile(self.datafile):
            with open(self.datafile, 'rb') as output:
                [self.images, self.labels, self.domains, self.files] = pickle.load(output)
        else:
            self.buildDataset()

    def buildDataset(self):
        offic31Folder = os.path.join(os.path.join(self.rootFolder, 'office31'), 'domain_adaptation_images')
        domainName = os.listdir(offic31Folder)

        for d in range(len(domainName)):
            domainFolder = os.path.join(os.path.join(offic31Folder, domainName[d]), 'images')
            classes = os.listdir(domainFolder)

            for c in range(len(classes)):
                classFolder = os.path.join(domainFolder, classes[c])
                imgFiles = os.listdir(classFolder)

                for i in range(len(imgFiles)):
                    img = Image.open( os.path.join(classFolder, imgFiles[i]) )
                    self.images.append(img)
                    self.labels.append(c)
                    self.domains.append(d)
                    self.files.append(os.path.join(classFolder, imgFiles[i]))

        with open(self.datafile, 'wb') as output:
            pickle.dump([self.images, self.labels, self.domains, self.files], output)
