from model import fc_resnet50, peak_response_mapping
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_size_h = 480
image_size_w = 720

def plt_result_pic(config, peak_response_maps, raw_img, response_maps, peak_responses, sites):
    if config['display_mode'] == 'response_map':
        num_plots = 2 + len(peak_response_maps)
        print('class map shape:', response_maps.shape)
        print(num_plots, ' numplots')
        print('Peak points:\n', peak_responses)

        f, axarr = plt.subplots(1, num_plots, figsize=(num_plots * 4, 4))
        img0 = Image.fromarray(np.asarray(raw_img))
        axarr[0].imshow(img0)
        axarr[0].axis('off')

        axarr[1].imshow(response_maps[0, 0].cpu(), interpolation='bicubic')
        axarr[1].axis('off')

        for idx, (prm, peak) in enumerate(sorted(zip(peak_response_maps, peak_responses), key=lambda v: v[-1][-1])):
            # ones = torch.ones_like(prm)
            # prm = torch.where((prm != 0), ones, prm)
            axarr[idx + 2].imshow(prm.cpu(), cmap=plt.cm.jet)
            axarr[idx + 2].axis('off')

    elif config['display_mode'] == 'rectangle':
        f, axarr = plt.subplots(1, 3, figsize=(3 * 4, 4))
        img0 = Image.fromarray(np.asarray(raw_img))
        axarr[0].imshow(img0)
        axarr[0].axis('off')

        axarr[1].imshow(response_maps[0, 0].cpu(), interpolation='bicubic')
        axarr[1].axis('off')

        axarr[2].imshow(img0)
        for site in sites:
            plt.gca().add_patch(
                plt.Rectangle((site[0], site[1]), site[2] - site[0],
                              site[3] - site[1], fill=False,
                              edgecolor='r', linewidth=1))

class Solver(object):

    def __init__(self, config):
        """Initialize configurations."""
        self.basebone = fc_resnet50(config)
        self.model = peak_response_mapping(config, self.basebone, **config['model'])
        self.config = config
        self.cuda = (config['cuda_device'] is not None)
        if self.cuda:
            print("Cuda is available?", config['cuda_device'])
            print(torch.cuda.is_available())
            self.model.to('cuda')

    @staticmethod
    def print_network(model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def dasr(self, input_var:[torch.tensor], img_w=None, img_h=None, raw_img=None, img_path=None, rate=1, config=None, dbscan=None):
        """Execute DASR."""
        self.model.eval()
        self.model.inference()

        visual_cues = self.model(input_var, img_w, img_h, img_path, peak_threshold=0, rate=rate, config=config, dbscan=dbscan)

        if visual_cues is None:
            print('No class peak response detected')
        elif config['plt_mode'] == 'show':
            response_maps, peak_responses, peak_response_maps, sites = visual_cues
            plt_result_pic(config, peak_response_maps, raw_img, response_maps, peak_responses, sites)
            plt.show()
        elif config['plt_mode'] == 'save':
            response_maps, peak_responses, peak_response_maps, sites = visual_cues
            plt_result_pic(config, peak_response_maps, raw_img, response_maps, peak_responses, sites)
            file_name = img_path.split('/')[-1].split('.')[0] + '_swav.jpg'
            plt.savefig('/media/data1/ybmiao/output/dasr/pic/swav/' + file_name)
            plt.close()
        elif config['plt_mode'] == 'none':
            pass
        else:
            raise ValueError('Wrong plt model select!')

    def test(self):
        model = self.model
        config = self.config
        self.print_network(model, config['model_name']['_name'])
