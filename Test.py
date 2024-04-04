import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import cv2
from net.TJNet import Network
from utils.data_val import test_dataset
import configparser

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

methods = []
methods.append('TJNet_best_camo')
methods.append('TJNet_best_chameleon')
methods.append('TJNet_best_cod10k')

data_names = []
data_names.append('CAMO')
data_names.append('CHAMELEON')
data_names.append('COD10K')
data_names.append('NC4K')

for method in methods:
    config = configparser.ConfigParser()
    config.read('./config.ini')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainsize', type=int,
                        default = config['Comm'].getint('trainsize'), help='testing size')
    parser.add_argument('--pth_root_path', type=str,
                        default=config['Test']['pth_root_path'])
    parser.add_argument('--test_dataset_path', type=str,
                        default=config['Test']['test_dataset_path'])
    parser.add_argument('--net_channel', type=int, default=config['Comm'].getint('net_channel'))
    opt = parser.parse_args()
    pth_path = opt.pth_root_path + method + ".pth"

    for _data_name in data_names:
        data_path = opt.test_dataset_path+'/{}/'.format(_data_name)
        save_path = './results/{}/{}/'.format(method, _data_name)
        os.makedirs(save_path, exist_ok=True)

        model = Network(opt.net_channel)
        
        # state_dict = torch.load(pth_path, map_location=torch.device('cpu'))
        # model.load_state_dict(state_dict, strict=False)
        
        model.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load(pth_path).items()})
        model.cuda()
        model.eval()

        image_root = '{}/Imgs/'.format(data_path)
        gt_root = '{}/GT/'.format(data_path)
        test_loader = test_dataset(image_root, gt_root, opt.trainsize)

        for i in range(test_loader.size):
            image, gt, name, _ = test_loader.load_data()
            print('> {} - {}'.format(_data_name, name))

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            _, _, res, e, c = model(image)

            res = F.interpolate(res, size=gt.shape,
                                mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path+name, res*255)
