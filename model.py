from typing import Union, Optional, List, Tuple, Dict, Iterable
import cv2
import numpy as np
from fnmatch import fnmatch
from peak_response_mapping import PeakResponseMapping
import torch
import torch.nn as nn
from torchvision import models
import torchvision


class DASR_ResNet(nn.Module):
    def __init__(self, model):
        super(DASR_ResNet, self).__init__()

        # feature encoding
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool2 = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        # self.dasr = model.layer4
        # self.dasr = nn.Sequential()
        self.dasr = nn.Sequential(*list(model.layer4.children())[:-2])
        self.pooling = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)
        # self.dasr = nn.Sequential(*list(model.layer4.children())[:-2])

        # self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)

        self.features_dasr = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool2,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            # self.dasr,
            # self.pooling,
        )


    def forward(self, x):
        x = self.features_dasr(x)

        target1 = np.asarray(x.detach().cpu())[:,:,0,0]
        target2 = np.asarray(x.detach().cpu())[:,:,0,1]

        x = torch.mean(x,1,keepdim=True)
        # x = self.pooling(x)
        return x


def fc_resnet50(config) -> nn.Module:
    """FC ResNet50."""
    if config['backbone'] == 'res50':
        # backbone = models.resnet50(pretrained=True)
        backbone = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        # backbone = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    elif config['backbone'] == 'swav':
        backbone = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    else:
        raise NotImplementedError

    dasr = DASR_ResNet(backbone)
    return dasr


def peak_response_mapping(
        config,
        backbone: nn.Module,
        enable_peak_stimulation: bool = True,
        enable_peak_backprop: bool = True,
        win_size: int = 3,
        sub_pixel_locating_factor: int = 1,
        filter_type: Union[str, int, float] = 'median') -> nn.Module:
    """Peak Response Mapping.
    """

    model = PeakResponseMapping(
        backbone,
        enable_ftr_save=config['feature_save'],
        feature_save_path=config['feature_save_path'],
        enable_feature_map_pooling=config['feature_map_pooling'],
        enable_peak_stimulation=enable_peak_stimulation,
        enable_peak_backprop=enable_peak_backprop,
        win_size=win_size,
        sub_pixel_locating_factor=sub_pixel_locating_factor,
        filter_type=filter_type)
    return model


if __name__ == "__main__":
    model = fc_resnet50(True)
    # model = fc_resnet50(False)
    print(model)

