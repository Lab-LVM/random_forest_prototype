from __future__ import print_function
import dataLoader.DA.genData
import os
import torch
from torch import nn
from torchvision.models import ResNet
from torchvision import transforms
#dependencies = ['torch']
BATCH_SIZE = 128


def buildDataset():
    o31 = dataLoader.DA.genData.office31(os.path.dirname(os.path.abspath(__file__)))


def getFeatures(dataset, backbone, crops):

    assert (crops > 0)
#    featreFile = backbone+str(crops) #here

def extractFeatures(dataset, backbone, crops):

    assert(crops > 0)

    model = torch.hub.load('pytorch/vision:v0.4.2', backbone, pretrained=True)

    if crops == 1:
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    assert(torch.cuda.is_available())
    model.to('cuda')

    modules = list(model.children())[: getLayerIndex(backbone)]
    seq = nn.Sequential(*modules)

    with torch.no_grad():

        for c in range(crops):
            print( "Feature extraction: " + str(c+1) + " of " + str(crops) + " crop starts!" )
            for i in range(len(dataset.images)):
                inputTensor = preprocess(dataset.images[i])
                inputTensor = inputTensor.unsqueeze(0)

                if (i+1) % BATCH_SIZE == 1:
                    inputBatch = inputTensor
                else:
                    inputBatch = torch.cat((inputBatch, inputTensor), 0)

                if (i + 1) % BATCH_SIZE == 0:
                    inputBatch = inputBatch.to('cuda')
                    f = seq(inputBatch)
                    f = f.unsqueeze(0)
                    if (i + 1) == BATCH_SIZE:
                        fs = f
                    else:
                        fs = torch.cat( (fs, f) , 0)

        fs = fs.unsqueeze(0)
        if c == 0:
            features = fs
        else:
            features = torch.cat( (features, fs) , 0)

    return features.cpu().detach().numpy()



def getLayerIndex(backbone):
    indices = {
        "resnet18": -1,
        "alexnet": 0
    }
    return indices.get(backbone, "Invalid backbone")