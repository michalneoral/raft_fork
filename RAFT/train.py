from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


from RAFT.core.raft import RAFT
from RAFT import evaluate
import RAFT.core.datasets as datasets

from torch.utils.tensorboard import SummaryWriter

from RAFT.core.utils.timer import Timer

from RAFT.core.utils.flow_viz import flow_to_color

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000 # 5000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def create_writer_if_not_set(self):
        if self.writer is None:
            self.writer = SummaryWriter()

    def write_dict(self, results):
        self.create_writer_if_not_set()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def write_image_dict(self, results):
        self.create_writer_if_not_set()

        for key in results:
            self.writer.add_image(key, results[key], self.total_steps)

    def write_images(self, inputs):
        self.create_writer_if_not_set()

        for key in inputs:
            im = inputs[key]

            # grid = torchvision.utils.make_grid(im)
            # self.writer.add_image(key, grid, self.total_steps)
            if key == 'valid':
                data = im.type(torch.uint8)
                data = torch.unsqueeze(data, 1) * 255
                self.writer.add_images(key, data, dataformats='NCHW', global_step=self.total_steps)
            elif 'flow' in key:
                data = im.detach().cpu().numpy().transpose(0,2,3,1)
                color_list = []
                for i in range(data.shape[0]):
                    color_list.append(flow_to_color(data[i, :, :, :]))
                    color_image = np.stack(color_list, axis=0)
                self.writer.add_images(key, color_image, dataformats='NHWC', global_step=self.total_steps)
            else:
                self.writer.add_images(key, im.type(torch.uint8), dataformats='NCHW', global_step=self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    train_timer = Timer()

    os.environ["CUDA_VISIBLE_DEVICES"] =  ",".join([str(gpu_n) for gpu_n in args.gpus])

    args.gpus = range(len(args.gpus))
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)


    should_keep_training = True
    print('Training...', train_timer.iter(), train_timer())
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            # print(total_steps, train_timer.iter(), train_timer())

            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)            

            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if (total_steps == 7) or (total_steps % VAL_FREQ == VAL_FREQ - 1):
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                # print('before val: ', train_timer.iter(), train_timer())
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                        # print('val: ', train_timer.iter(), train_timer())
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                        # print('val: ', train_timer.iter(), train_timer())
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))
                        # print('val: ', train_timer.iter(), train_timer())
                    elif val_dataset == 'viper':
                        results.update(evaluate.validate_viper(model.module))
                        # print('val: ', train_timer.iter(), train_timer())
                print('after val: ', total_steps, train_timer.iter(), train_timer())

                logger.write_dict(results)

                logger.write_images({'image1': image1, 'image2': image2, 'valid': valid})
                logger.write_images({'flow_gt': flow})
                logger.write_images({'flow_est': flow_predictions[-1]})

                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAFT PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--dashcam_augmenentation', action='store_true')
    parser.add_argument('--blend_source', default='/datagrid/public_datasets/COCO/train2017', help="path to blending images")


    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)