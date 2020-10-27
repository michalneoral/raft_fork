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

from core.utils.utils import InputPadder
from core.utils import frame_utils

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img_torch = torch.from_numpy(img).permute(2, 0, 1).float()
    return img_torch[None].to(DEVICE), img

@torch.no_grad()
def gen(args):

    for dataset_name in ['training', 'testing']:

        data_root = '/datagrid/public_datasets/KITTI/multiview/{:s}/image_2'.format(dataset_name)
        save_root = '/datagrid/personal/neoral/datasets/optical_flow_neomoseg/raft_new_export/kitti/{:s}'.format(dataset_name)

        ITERS = 24

        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))

        model = model.module
        model.to(DEVICE)
        model.eval()

        model_e = model
    
        for t_scale_i in range(5):
            t_scale = t_scale_i + 1
            pbar = tqdm(range(200))
            for sequence in pbar:
                for image_n in range(21):
                    path_im1 = os.path.join(data_root, '{:06d}_{:02d}.png'.format(sequence, image_n))
                    path_im2 = os.path.join(data_root, '{:06d}_{:02d}.png'.format(sequence, image_n + t_scale))
                    
                    if not os.path.exists(path_im1) or not os.path.exists(path_im2):
                        continue
                    
                    pbar.set_description('t_scale = {:d}: {:06d}: {:02d} and {:02d}'.format(t_scale, sequence, image_n, image_n + t_scale))
    
                    with torch.no_grad():
                        # kitti images
                        image1, image1_orig = load_image(path_im1)
                        image2, image2_orig = load_image(path_im2)

                        padder = InputPadder(image1.shape, mode='kitti')
                        image1, image2 = padder.pad(image1.cuda(), image2.cuda())

                        _, flow_pr = model_e(image1, image2, iters=ITERS, test_mode=True)
                        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

                        #output_filename = os.path.join(save_root, 'time_scale_{:d}'.format(t_scale), 'forward'), '{:06d}_{:02d}.png'.format(sequence, image_n)
                        #frame_utils.writeFlowKITTI(output_filename, flow)

                        #flow_predictions = model(image1, image2, iters=16, test_mode=True)
                        flow_gen.save_outputs(image1_orig, image2_orig, flow, os.path.join(save_root, 'time_scale_{:d}'.format(t_scale), 'forward'), '{:06d}_{:02d}.png'.format(sequence, image_n))

                        #flow_predictions = model(image2, image1, iters=16, test_mode=True)
                        _, flow_pr = model_e(image2, image1, iters=ITERS, test_mode=True)
                        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
                        flow_gen.save_outputs(image2_orig, image1_orig, flow, os.path.join(save_root, 'time_scale_{:d}'.format(t_scale), 'backward'), '{:06d}_{:02d}.png'.format(sequence, image_n + t_scale))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    args = parser.parse_args()
    gen(args)