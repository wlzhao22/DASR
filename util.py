import argparse
import yaml
import numpy as np
import math
import logging
from logging import handlers
import torch


class Logger(object):
    """
    example:
        import time
        logpath = './log.txt'
        log = Logger(logpath, level='info')
        log.logger.info('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    """
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self,filename,printflag=False,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        if printflag:
            sh = logging.StreamHandler()
            sh.setFormatter(format_str)
            self.logger.addHandler(sh)
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(th)


# YAML should override the argparser's content
def _parse_args_and_yaml(given_parser=None):
    parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument(
        '-c',
        '--config_yaml',
        default='your_path',
        type=str,
        metavar='FILE',
        help='YAML config file specifying default arguments')
    if given_parser is None:
        given_parser = parser
    given_configs, remaining = given_parser.parse_known_args()
    if given_configs.config_yaml:
        with open(given_configs.config_yaml, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            given_parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = given_parser.parse_args(remaining)
    return args

    # # Cache the args as a text string to save them in the output dir later
    # args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    # return args, args_text


def calculate_resized_w_h(w, h, max_length=512):
    if w > h:
        rate = w / max_length
        w_resized = max_length
        h_resized = int(h / rate)
    else:
        rate = h / max_length
        w_resized = int(w / rate)
        h_resized = max_length
    return w_resized, h_resized, rate


def calculate_mean_std(raw_img):
    img = np.array(raw_img).astype(np.float32) / 255.
    mean = []
    std = []
    for i in range(3):
        pixels = img[:, :, i].ravel()
        mean.append(np.mean(pixels))
        std.append(np.std(pixels))


def nms(dets, thres):
    '''
    https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    :param dets:  [[x1,y1,x2,y2,score], [x1,y1,x2,y2,score],,,]
    :param thres: for example 0.5
    :return: the rest ids of dets
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = dets[:, 4]
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(xx2 - xx1 + 1.0, 0.0)
        h = np.maximum(yy2 - yy1 + 1.0, 0.0)

        inters = w * h
        unis = areas[i] + areas[order[1:]] - inters
        ious = inters / unis

        inds = np.where(ious <= thres)[0]  # return the rest boxxes whose iou<=thres

        order = order[
            inds + 1]  # for exmaple, [1,0,2,3,4] compare '1', the rest is 0,2 who is the id, then oder id is 1,3

    return keep


def estimate_ellipse(y, x, img_shape, peak_response_map=None):
    """
    :param y: np.array
    :param x: np.array
    :param img_shape: (h, w)
    :return:
    """
    y = y.cpu().numpy()
    x = x.cpu().numpy()
    above_points = np.stack([y, x], axis=-1)
    above_points_mu = np.mean(above_points, axis=0)
    above_points_minus_mu = above_points - above_points_mu
    cov = np.matmul(above_points_minus_mu.transpose(), above_points_minus_mu)
    cov = cov / np.shape(above_points_minus_mu)[0]
    values, vectors = np.linalg.eig(cov)
    values = np.eye(values.shape[0]) * values
    # values = 2.5 * np.sqrt(values)
    values = 2 * np.sqrt(values)
    angle = math.atan2(cov[1, 0] + cov[0, 1], cov[1, 1] - cov[0, 0]) / 2 / math.pi * 180
    minor_axis, major_axis = np.sort(values.flatten())[-2:]
    angle_radius = angle / 180.0 * np.pi
    ux = major_axis * np.cos(angle_radius)
    uy = major_axis * np.sin(angle_radius)
    vx = minor_axis * np.cos(angle_radius + np.pi / 2.)
    vy = minor_axis * np.sin(angle_radius + np.pi / 2.)
    half_width = np.sqrt(ux ** 2 + vx ** 2)
    half_height = np.sqrt(uy ** 2 + vy ** 2)
    y_min, y_max = np.clip(
        np.array([-half_height, half_height], dtype=np.int32) + int(above_points_mu[0]), 0, img_shape[0] - 1)
    x_min, x_max = np.clip(
        np.array([-half_width, half_width], dtype=np.int32) + int(above_points_mu[1]), 0, img_shape[1] - 1)
    return np.array([x_min, y_min, x_max, y_max])
