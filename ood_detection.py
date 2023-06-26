# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Authors: Vivek Narayanaswamy. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import os
import sys
sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./'))
import time

import argparse
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score, accuracy_score
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models

from datasets import get_loaders
import hparams_registry

import matplotlib.pyplot as plt
from scipy import misc
from utils.cal_metric import metric
from torchvision import models

# Global Variable for saving in_dist_scores
in_scores = []

def parse_arguments():
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
    parser.add_argument('--in_dataset', default='bloodmnist', type=str, help='Name of the In dist. Dataset')
    parser.add_argument('--out_test_dataset', default='dermamnist', type=str, help='Name of the test OOD Dataset [nct, ISIC2019_rem_classes]')
    parser.add_argument('--model_type', default='wrn', type=str, help='Model type : resnet50, wrn')
    parser.add_argument('--T', default=1.0, type=float, help='Temperature for the Energy function')
    parser.add_argument('--m_in', default=-20.0, type=float, help='Margin for In-dataset')
    parser.add_argument('--m_out', default=-7.0, type=float, help='Margin for Out-dataset')
    parser.add_argument('--dir_name', default='in_latent_out_pix', type=str, help='Name of the dir to store results')

    # WRN Architecture
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')

    parser.set_defaults(argument=True)
    args = parser.parse_args()

    return args


class OODManager(object):
    """
    Class to manage OOD Detection
    :param model: name of classifier model
    :param in_params: in_data parameters
    :param DataLoader in_loader: iterate through in_data validation data
    :param DataLoader ood_loader: iterate through the OOD data
    :param dict config: dictionary with hyperparameters
    :return: object of OODManager class
    """

    def __init__(self, model=None, in_params={}, in_loader=None, ood_loader=None, args=None):
        self.model = model          # Pre-trained with loaded weights. Model is in eval mode
        self.params = in_params
        self.device = self.params['device']
        self.in_loader = in_loader
        self.ood_loader = ood_loader
        self.args = args
        self.softmax = nn.Softmax(dim=1)

        # Helper functions
        self.to_np = lambda x: x.data.cpu().numpy()
        self.concat = lambda x: np.concatenate(x, axis=0)


    def compute_scores(self, images):

        # Importance of sign: We want higher scores for in_dist samples while lower_scores for ood samples
        # We alter the signs to be consistent with the metric computation code which considers
        # p(samples) > delta -> 1 : In dist
        # and p(samples) < delta -> 0 : OOD

        outputs, _ = self.model(images.to(self.device))
        scores = self.to_np((self.args.T * torch.logsumexp(outputs / self.args.T, dim=1)))  # Negative Energy Score
        scores = np.reshape(scores, -1)
        return scores


    def eval_ood_performance(self):

        # Creating a directory where the scores will be saved
        save_dir = os.path.join(self.params['result_dir'], self.args.model_type, self.args.in_dataset, self.args.dir_name, self.args.out_test_dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        start = time.time()

        # Files to save the OOD scores
        ood_method = 'energy'
        g1 = open(os.path.join(save_dir, "scores_{}_In.txt".format(ood_method)), 'w')
        g2 = open(os.path.join(save_dir, "scores_{}_Out.txt".format(ood_method)), 'w')

        # Total number of examples in the in_dist validation data
        N = len(self.in_loader.dataset)
        print('No. of in-dist samples = {}'.format(N))

        ########################################In-distribution###########################################
        print("Processing in-distribution data")
        t0 = time.time()
        count = 0
        global in_scores
        if len(in_scores) == 0:
            print('Computing in_dist scores')
            for j, data in enumerate(self.in_loader):
                images, _ = data
                batch_size = images.shape[0]

                if count + batch_size > N:
                    images = images[:N-count]
                    batch_size = images.shape[0]

                in_scores.append(self.compute_scores(images))

                count += batch_size
                print("{:4}/{:4} In dist.images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
                t0 = time.time()

                if count == N: break

            in_scores = self.concat(in_scores).copy()

            for k in range(count):
                g1.write("{}\n".format(in_scores[k]))

            print(in_scores.shape)
            print('Saved in_dist scores')

        else:
            for k in range(len(in_scores)):
                g1.write("{}\n".format(in_scores[k]))

            print(in_scores.shape)
            print('Already computed in_dist_scores')
            print('Saved in_dist_scores')


        ###################################Out-of-Distributions#####################################
        t0 = time.time()
        print("Processing out-of-distribution data")
        count = 0
        out_scores = []
        N = len(self.ood_loader.dataset)
        for j, data in enumerate(self.ood_loader):
            images, _ = data
            batch_size = images.shape[0]

            if count + batch_size > N:
                images = images[:N-count]
                batch_size = images.shape[0]

            out_scores.append(self.compute_scores(images))
            count += batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
            t0 = time.time()

            if count== N: break

        out_scores = self.concat(out_scores).copy()

        for k in range(count):
            g2.write("{}\n".format(out_scores[k]))

        g1.close()
        g2.close()

        # Compute and Display Metrics
        results = metric(save_dir, [ood_method])
        self.print_results(results, [ood_method])


    def print_results(self, results, stypes):
        mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

        print('In_distribution: ' + self.args.in_dataset)
        print('Out_distribution: '+ self.args.out_test_dataset)
        print('')

        for stype in stypes:
            print('OOD detection method: ' + stype)
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*results[stype]['FPR']), end='')
            print(' {val:6.2f}'.format(val=100.*results[stype]['DTERR']), end='')
            print(' {val:6.2f}'.format(val=100.*results[stype]['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*results[stype]['AUIN']), end='')
            print(' {val:6.2f}\n'.format(val=100.*results[stype]['AUOUT']), end='')
            print('')
#############################################################################################################

def main():

    # Get the hyper-parameters as well as the settings for 'in_dataset' & 'ood_dataset'
    in_dataset_params = hparams_registry.default_hparams(args.in_dataset)
    in_dataset_params['augment_type'] = 'none'
    in_dataset_params['batch_size'] = 256
    in_dataset_params['num_workers'] = 4

    if args.model_type == 'wrn':
        in_dataset_params['im_size'] =32

    out_test_dataset_params = hparams_registry.default_hparams(args.out_test_dataset)
    out_test_dataset_params['augment_type'] = 'none'
    out_test_dataset_params['batch_size'] = 256
    out_test_dataset_params['num_workers'] = in_dataset_params['num_workers']
    out_test_dataset_params['im_size'] = in_dataset_params['im_size']

    if args.in_dataset == 'bloodmnist' and args.out_test_dataset == 'bloodmnist':
        out_test_dataset_params['val_csv'] = os.path.join(out_test_dataset_params['csv_dir'], 'bloodmnist_out.csv')
        print('Novel classes for {}'.format(args.in_dataset))

    if args.in_dataset == 'dermamnist' and args.out_test_dataset == 'dermamnist':
        out_test_dataset_params['val_csv'] = os.path.join(out_test_dataset_params['csv_dir'], 'dermamnist_out.csv')
        print('Novel classes for {}'.format(args.in_dataset))

    if args.in_dataset == 'pathmnist' and args.out_test_dataset == 'pathmnist':
        out_test_dataset_params['val_csv'] = os.path.join(out_test_dataset_params['csv_dir'], 'pathmnist_out.csv')
        print('Novel classes for {}'.format(args.in_dataset))

    if args.in_dataset == 'tissuemnist' and args.out_test_dataset == 'tissuemnist':
        out_test_dataset_params['val_csv'] = os.path.join(out_test_dataset_params['csv_dir'], 'tissuemnist_out.csv')
        print('Novel classes for {}'.format(args.in_dataset))

    if args.in_dataset == 'organcmnist' and args.out_test_dataset == 'organcmnist':
        out_test_dataset_params['val_csv'] = os.path.join(out_test_dataset_params['csv_dir'], 'organcmnist_out.csv')
        print('Novel classes for {}'.format(args.in_dataset))

    if args.in_dataset == 'organamnist' and args.out_test_dataset == 'organamnist':
        out_test_dataset_params['val_csv'] = os.path.join(out_test_dataset_params['csv_dir'], 'organamnist_out.csv')
        print('Novel classes for {}'.format(args.in_dataset))

    if args.in_dataset == 'organsmnist' and args.out_test_dataset == 'organsmnist':
        out_test_dataset_params['val_csv'] = os.path.join(out_test_dataset_params['csv_dir'], 'organsmnist_out.csv')
        print('Novel classes for {}'.format(args.in_dataset))

    if args.in_dataset == 'octmnist' and args.out_test_dataset == 'octmnist':
        out_test_dataset_params['val_csv'] = os.path.join(out_test_dataset_params['csv_dir'], 'octmnist_out.csv')
        print('Novel classes for {}'.format(args.in_dataset))

    ############################################################################

    # CUDA / CPU
    if torch.cuda.is_available():
        in_dataset_params['device'] = 'cuda'
        out_test_dataset_params['device'] = 'cuda'
        print('Experiment Running on cuda')

    else:
        in_dataset_params['device'] = 'cpu'
        out_test_dataset_params['device'] = 'cpu'
        print('Experiment Running on cpu')
    ############################################################################

    # Val Loader for 'in_dataset'
    _, in_loader = get_loaders(args.in_dataset, in_dataset_params, distributed=False)

    # Loader for 'out_dataset'
    _, out_loader = get_loaders(args.out_test_dataset, out_test_dataset_params, distributed=False)
    print('Loaded In_dist and OOD Loaders')

    ############################################################################
    if args.model_type == 'wrn':
        from modeldefs.mnist_wrn import MNISTWideResNet
        model = MNISTWideResNet(args.layers, in_dataset_params['num_classes'], args.widen_factor, dropRate=args.droprate)
        print('Chosen model = {}'.format(args.model_type))
    else:
        raise NotImplementedError

    # Model Checkpoint File
    model_checkpoint_file = os.path.join(in_dataset_params['checkpoint_dir'], args.model_type, args.in_dataset, args.dir_name, f'in_latent_out_pix_m_in_{args.m_in}-m_out_{args.m_out}-T_{args.T}-ckpt_last.pth.tar')
    checkpoint = torch.load(model_checkpoint_file)
    print(model_checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded model with saved weights and biases')
    print('OOD Metric : energy')
    print(checkpoint['epoch'])

    ############################################################################
    if(torch.cuda.device_count() > 1):
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        model = model.to(in_dataset_params['device'])
        model.eval()

    else:
        model = model.to(in_dataset_params['device'])
        model.eval()

    ############################################################################

    manager = OODManager(model, in_dataset_params, in_loader, out_loader, args)
    manager.eval_ood_performance()



if __name__ == '__main__':
    start = time.time()
    out_test_dataset_list = ['bloodmnist', 'dermamnist', 'octmnist', 'organamnist', 'organcmnist', 'organsmnist', 'pathmnist', 'tissuemnist']
    for name in out_test_dataset_list:
        args = parse_arguments()
        args.out_test_dataset = name
        print(args)
        main()
    end = time.time()
    time_elapsed = end - start
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
