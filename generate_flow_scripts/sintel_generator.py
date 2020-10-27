import sys
sys.path.append("/datagrid/personal/neoral/repos/raft_debug")
sys.path.append('core')

import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import core.datasets as datasets
from core.utils import flow_viz
from core.raft import RAFT
import core.utils.flow_gen as flow_gen
from core.utils.flow_viz import flow_to_image
from tqdm import tqdm
from tqdm.auto import trange
from time import sleep


DEVICE = 'cuda'

def gen(args):

    for dataset_name in ['training', 'test']:
        for dataset_subname in ['clean', 'final']:

            data_root = '/datagrid/public_datasets/Sintel-complete/{:s}/{:s}'.format(dataset_name, dataset_subname)
            save_root = '/datagrid/personal/neoral/tmp/raft_export/sintel/{:s}_{:s}'.format(dataset_name, dataset_subname)

            model = RAFT(args)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(args.model))

            model.to(DEVICE)
            model.eval()

            for t_scale_i in range(5):
                t_scale = t_scale_i + 1
                pbar = tqdm(os.listdir(data_root)[0:2])
                for sequence in pbar:
                    for image_n in range(51):
                        path_im1 = os.path.join(data_root, '{:s}/frame_{:04d}.png'.format(sequence, image_n))
                        path_im2 = os.path.join(data_root, '{:s}/frame_{:04d}.png'.format(sequence, image_n + t_scale))

                        if not os.path.exists(path_im1) or not os.path.exists(path_im2):
                            continue

                        pbar.set_description('t_scale = {:d}: {:s}/{:02d} and {:02d}'.format(t_scale, sequence, image_n, image_n + t_scale))

                        with torch.no_grad():
                            # kitti images
                            image1 = flow_gen.load_image(path_im1)
                            image2 = flow_gen.load_image(path_im2)
                            flow_predictions = model.module(image1, image2, iters=16)
                            flow_gen.save_outputs(image1[0], image2[0], flow_predictions[-1][0], os.path.join(save_root, 'time_scale_{:d}'.format(t_scale), 'forward'), '{:s}/frame_{:04d}.png'.format(sequence, image_n))

                            flow_predictions = model.module(image2, image1, iters=16)
                            flow_gen.save_outputs(image2[0], image1[0], flow_predictions[-1][0], os.path.join(save_root, 'time_scale_{:d}'.format(t_scale), 'backward'), '{:s}/frame_{:04d}.png'.format(sequence, image_n + t_scale))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--iters', type=int, default=12)

    args = parser.parse_args()
    gen(args)