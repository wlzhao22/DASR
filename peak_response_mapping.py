from types import MethodType
import cv2
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import math
from matplotlib import pyplot as plt
import copy
from peak_stimulation import peak_stimulation
from peak_backprop import pr_conv2d
import os
from util import nms, estimate_ellipse

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation

class PeakResponseMapping(nn.Sequential):

    def __init__(self, *args, **kargs):
        super(PeakResponseMapping, self).__init__(*args)
        self.inferencing = False
        # use global average pooling to aggregate responses if peak stimulation is disabled
        self.enable_peak_stimulation = kargs.get('enable_peak_stimulation', True)
        # return only the class response maps in inference mode if peak backpropagation is disabled
        self.enable_peak_backprop = kargs.get('enable_peak_backprop', True)
        # window size for peak finding
        self.win_size = kargs.get('win_size', 3)
        # sub-pixel peak finding
        self.sub_pixel_locating_factor = kargs.get('sub_pixel_locating_factor', 1)
        self.filter_type = 'median'
        self.peak_filter = self._over_mean_filter

        self.enable_ftr_save = kargs.get('enable_ftr_save', False)
        if self.enable_ftr_save:
            out_ftr_path = kargs.get('feature_save_path', None)
            if out_ftr_path is not None:
                self.file = open(out_ftr_path,'a')
            else:
                raise ValueError("Wrong output path!")

    @staticmethod
    def _median_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.median(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _mean_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _over_mean_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2) * 1
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _max_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.max(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    def _patch(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module._original_forward = module.forward
                module.forward = MethodType(pr_conv2d, module)

    def _recover(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, '_original_forward'):
                module.forward = module._original_forward

    def forward(self, input, img_w=None, img_h=None, img_path=None, peak_threshold=0, rate=1, config=None, dbscan=None):
        assert input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'
        assert config is not None, 'No config yml input.'


        input.requires_grad_()

        # classification network forwarding
        class_response_maps = super(PeakResponseMapping, self).forward(input)



        if config['use_DASR*']:
            peak_list, score_list = peak_stimulation(class_response_maps, return_aggregation=False,
                                                     win_size=self.win_size)

            score_list = score_list.detach().numpy()
            idxs = range(peak_list.size(0))

            # extract instance-aware visual cues, i.e., peak response maps
            assert class_response_maps.size(
                0) == 1, 'Currently inference mode (with peak backpropagation) only supports one image at a time.'

            peak_response_maps = []
            valid_peak_list = []
            sites = []

            # peak_threshold = np.percentile(class_response_maps.detach().cpu(), 70)
            peak_threshold = 0

            # peak backpropagation
            grad_output = class_response_maps.new_empty(class_response_maps.size())
            boxes = []
            for idx in idxs:
                peak_val = class_response_maps[
                    peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]]
                if peak_val > peak_threshold:
                    grad_output.zero_()
                    # starting from the peak
                    grad_output[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]] = 1

                    if input.grad is not None:
                        input.grad.zero_()
                    class_response_maps.backward(grad_output, retain_graph=True)
                    prm = input.grad.detach().sum(1).clone().clamp(min=0)
                    # print('prm  ', prm)

                    target = np.asarray(input.grad.detach().cpu().clone())

                    zero = torch.zeros_like(prm)
                    ts = np.percentile(np.array(prm.detach().cpu()), 95)
                    prm = torch.where((prm < ts), zero, prm)

                    peak_response_maps.append(prm / prm.sum())
                    valid_peak_list.append(peak_list[idx, :])

                    prm_points = torch.nonzero(prm)
                    x = prm_points[:, 1]
                    y = prm_points[:, 2]
                    site = estimate_ellipse(x, y, (img_h, img_w), peak_response_maps[-1])

                    for i in range(len(site)):
                        site[i] = int(site[i] * rate)

                    site_idx = np.append(site, score_list[idx])
                    boxes.append(site_idx)

            boxes = np.array(boxes)
            boxes_list = nms(boxes, 0.3)
            for idx in range(len(boxes_list)):
                site = boxes[boxes_list[idx],:-1].astype(int)
                info = img_path + ">" + str(idx).rjust(4, '0') + " " + str(site)[1:-1] + "\n"

                if self.enable_ftr_save:
                    self.file.writelines(info)
                else:
                    print(info)
                    sites.append(site)

        else:
            peak_list, score_list = peak_stimulation(class_response_maps, return_aggregation=False,
                                                     win_size=self.win_size,
                                                     peak_filter=self.peak_filter(class_response_maps))

            # peak_filter = self.peak_filter(input))
            nms_list = torch.cat((peak_list, score_list.view(-1, 1)), dim=1)
            if config['use_nms']:
                if len(nms_list) == 0:
                    idxs = []
                else:
                    nms_list = np.asarray(nms_list.detach())
                    idxs = nms(nms_list, 0.3)
            else:
                idxs = range(peak_list.size(0))

            # extract instance-aware visual cues, i.e., peak response maps
            assert class_response_maps.size(
                0) == 1, 'Currently inference mode (with peak backpropagation) only supports one image at a time.'

            peak_response_maps = []
            valid_peak_list = []
            sites = []

            peak_threshold = 0
            # peak backpropagation
            grad_output = class_response_maps.new_empty(class_response_maps.size())
            for idx in idxs:
                peak_val = class_response_maps[
                    peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]]
                if peak_val > peak_threshold:
                    # print('k ')
                    grad_output.zero_()
                    # starting from the peak
                    grad_output[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]] = 1

                    if input.grad is not None:
                        input.grad.zero_()
                    class_response_maps.backward(grad_output, retain_graph=True)
                    prm = input.grad.detach().sum(1).clone().clamp(min=0)
                    # print('prm  ', prm)

                    target = np.asarray(input.grad.detach().cpu().clone())

                    zero = torch.zeros_like(prm)
                    ts = np.percentile(np.array(prm.detach().cpu()), 95)
                    prm = torch.where((prm < ts), zero, prm)


                    peak_response_maps.append(prm / prm.sum())
                    valid_peak_list.append(peak_list[idx, :])


                    prm_points = torch.nonzero(prm)
                    x = prm_points[:, 1]
                    y = prm_points[:, 2]
                    site = estimate_ellipse(x, y, (img_h, img_w), peak_response_maps[-1])

                    for i in range(len(site)):
                        site[i] = int(site[i] * rate)

                    info = img_path + ">" + str(idx).rjust(4, '0') + " " + str(site)[1:-1] + "\n"

                    if self.enable_ftr_save:
                        self.file.writelines(info)
                    else:
                        print(info)
                        sites.append(site)

        class_response_maps = class_response_maps.detach()
        if len(peak_response_maps) > 0:
            valid_peak_list = torch.stack(valid_peak_list)
            peak_response_maps = torch.cat(peak_response_maps, 0)
            return class_response_maps, valid_peak_list, peak_response_maps, sites
        else:
            return None


    def train(self, mode=True):
        super(PeakResponseMapping, self).train(mode)
        if self.inferencing:
            self._recover()
            self.inferencing = False
        return self

    def inference(self):
        super(PeakResponseMapping, self).train(False)
        self._patch()
        self.inferencing = True
        return self
