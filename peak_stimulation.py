import torch
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np


class PeakStimulation(Function):

    @staticmethod
    def forward(ctx, input, return_aggregation, win_size, peak_filter):
        ctx.num_flags = 4

        # peak finding
        assert win_size % 2 == 1, 'Window size for peak finding must be odd.'

        offset = (win_size - 1) // 2
        padding = torch.nn.ConstantPad2d(offset, float('-inf'))
        # 给input加Padding
        padded_maps = padding(input)

        batch_size, num_channels, h, w = padded_maps.size()

        element_map = torch.arange(0, h * w).long().view(1, 1, h, w)[:, :, offset: -offset, offset: -offset]
        element_map = element_map.to(input.device)

        scores, indices = F.max_pool2d(
            padded_maps,
            kernel_size=win_size,
            stride=1,
            return_indices=True)
        peak_map = (indices == element_map)

        # peak filtering
        if peak_filter:
            input_filter = peak_filter
            mask = input >= input_filter
            peak_map = (peak_map & mask)


        # peak_list = torch.nonzero(peak_map)

        peak_list = torch.nonzero(peak_map).cpu().numpy()
        map_shape = input.shape
        del_idx = []
        for idx in range(peak_list.shape[0]):
            if peak_list[idx, 2] == 0 or peak_list[idx, 3] == 0 or peak_list[idx, 2] == map_shape[2] - 1 \
                    or peak_list[idx, 3] == map_shape[3] - 1:
                del_idx.append(idx)

        peak_list = torch.from_numpy(np.delete(peak_list, del_idx, axis=0))

        score_list = []
        for i in peak_list:
            score_list.append(scores[i[0]][i[1]][i[2]][i[3]])

        # ctx.mark_non_differentiable(peak_list)
        peak_map = peak_map.float()
        ctx.mark_non_differentiable(peak_map)

        # peak aggregation
        if return_aggregation:
            ctx.save_for_backward(input, peak_map)
            return peak_list, (input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
                   peak_map.view(batch_size, num_channels, -1).sum(2)
        else:
            ctx.save_for_backward(input, peak_map)
            return peak_list, torch.Tensor(score_list)

    @staticmethod
    def backward(ctx, grad_peak_list, grad_output):
        input, peak_map, = ctx.saved_tensors
        batch_size, num_channels, _, _ = input.size()
        grad_input = peak_map * grad_output.view(batch_size, num_channels, 1, 1)
        return (grad_input,) + (None,) * ctx.num_flags


def peak_stimulation(input, return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation.apply(input, return_aggregation, win_size, peak_filter)
