import os
from datasets import image_transform, dasr_img_transform
from solver import Solver
import yaml
import PIL.Image
from util import calculate_resized_w_h, calculate_mean_std, Logger
from sklearn.cluster import DBSCAN
import numpy as np
import time
from tqdm import tqdm

def get_path_list_from_file(file_path):
    imgs_file = open(file_path, "r")
    img_paths = []
    while True:
        img_path = imgs_file.readline().strip()
        if not img_path:
            break
        img_paths.append(img_path)
    return img_paths

def get_file_path(configs):
    """
    根据config的参数 选择数据集文件路径
    :param configs:
    :return:
    """
    if configs['model_select'] == 'ref':
        if configs['dataset'] == 'Ins160':
            file_path = configs['Ins160_path']
        elif configs['dataset'] == 'Ins335':
            file_path = configs['Ins335_path']
        elif configs['dataset'] == 'INSTRE':
            file_path = configs['INSTRE_path']
        else:
            print("Wrong Dataset Select!")
            exit()
    elif configs['model_select'] == 'qry':
        if configs['dataset'] == 'Ins160':
            file_path = configs['Ins160_qry_path']
        else:
            print("Wrong Dataset Select!")
            exit()
    else:
        print("Wrong Dataset Select!")
        exit()

    return file_path

def img_transform(configs, raw_img, h_resized, w_resized):
    if configs['transform_mode'] == 'Ins160':
        img_trans = image_transform(image_size=[h_resized, w_resized], mean=[0.45950904, 0.45434573, 0.44193703],
                                    std=[0.25827372, 0.25901473, 0.26378572])
        input_var = img_trans(raw_img).unsqueeze(0).cuda().requires_grad_()
    elif configs['transform_mode'] == 'ImageNet':
        img_trans = image_transform(image_size=[h_resized, w_resized], mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
        input_var = img_trans(raw_img).unsqueeze(0).cuda().requires_grad_()
    elif configs['transform_mode'] == 'single':
        mean, std = calculate_mean_std(raw_img)
        img_trans = image_transform(image_size=[480, 720], mean=mean, std=std)
        input_var = img_trans(raw_img).unsqueeze(0).cuda().requires_grad_()
    elif configs['transform_mode'] == 'instre':
        img_trans = image_transform(image_size=[h_resized, w_resized], mean=[0.4601, 0.4399, 0.4119],
                                    std=[0.2376, 0.2310, 0.2387])
        input_var = img_trans(raw_img).unsqueeze(0).cuda().requires_grad_()
    elif configs['transform_mode'] == 'Ins335':
        img_trans = image_transform(image_size=[h_resized, w_resized], mean=[0.4623, 0.4593, 0.4469],
                                    std=[0.2142, 0.2123, 0.2164])
        input_var = img_trans(raw_img).unsqueeze(0).cuda().requires_grad_()
    elif configs['transform_mode'] == 'DASR_hcxiao':
        img_trans = dasr_img_transform(image_size=[h_resized, w_resized],
                                       mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225]
                                       )
        input_var = img_trans(raw_img).unsqueeze(0).cuda().requires_grad_()
    else:
        print("Error transform mode in config.yml!")
        exit(-1)
    return input_var

def detect_single_picture(img_path, solver, dbscan=None):
    raw_img = PIL.Image.open(img_path).convert('RGB')

    w_resized, h_resized, rate = calculate_resized_w_h(raw_img.size[0], raw_img.size[1])

    # image_transform
    input_var = img_transform(config, raw_img, h_resized, w_resized)

    solver.dasr(input_var, w_resized, h_resized, raw_img, img_path, rate=rate, config=config, dbscan=dbscan)

def main(config=None):
    if config is None:
        with open("configs/config.yml", 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    solver = Solver(config)
    dbscan = DBSCAN(eps=2, min_samples=1)

    if config['detect_mode'] == 'single':
        img_path = config['img_path']
        detect_single_picture(img_path, solver, dbscan=dbscan)
    elif config['detect_mode'] == 'test':
        solver.test()  # output model structure
    elif config['detect_mode'] == 'dataset':
        file_path = get_file_path(config) # get file path
        img_paths = get_path_list_from_file(file_path) # get file path list
        for img_path in tqdm(img_paths):
            detect_single_picture(img_path, solver, dbscan=dbscan)
    else:
        print("Error detect mode in config.yml!")
        return


if __name__ == "__main__":
    with open("configs/dasr.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_device']
    main(config)
