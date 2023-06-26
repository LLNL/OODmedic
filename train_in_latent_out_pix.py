#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Authors: Vivek Narayanaswamy. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

## 'Training with Calibration Protocol: Latent Space Inliers and Pixel Space Outliers'

import os
import sys
import time
import argparse
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('./'))

from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score, accuracy_score
import numpy as np
import pickle

import torch
from torchvision.transforms import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from utils.util import *
from datasets import *
from torchvision import models
import hparams_registry


class In_Dist_Neg_Aug(torch.utils.data.Dataset):
    def __init__(self, in_dataset, csvfile, cfg, transform, device, aug_list=[], use_cache=False):
        self.in_dataset = in_dataset
        print('NDA for {}'.format(self.in_dataset))
        self.csvfile = csvfile
        self.cfg = cfg # config file
        self.datadir = self.cfg['data_dir']
        self.transform = transform

        # MNIST
        if 'mnist' in self.in_dataset:
            data = pd.read_csv(self.csvfile)
            self.indices = data['index'].values
            self.labels = data['labels'].values
            le = preprocessing.LabelEncoder()
            self.labels = le.fit_transform(self.labels)
            with open(os.path.join(self.cfg['data_dir'], self.in_dataset+'_combined.pkl'), 'rb+') as f:
                tmp = pickle.load(f)
            f.close()
            self.images = tmp['images']
        else:
            raise NotImplementedError

        self.device = device
        self.use_cache = use_cache
        self.cache = dict()
        self.aug_list = aug_list
        print('Augmentations chosen {}'.format(self.aug_list))

    def __getitem__(self, idx):
        if idx in self.cache and self.use_cache:
            return self.cache[idx]

        if self.in_dataset == 'bloodmnist' or self.in_dataset == 'pathmnist' or self.in_dataset == 'dermamnist':
            img = self.images[self.indices[idx]]

        elif self.in_dataset == 'tissuemnist' or self.in_dataset == 'organcmnist' or self.in_dataset == 'organamnist' or self.in_dataset == 'organsmnist' or self.in_dataset == 'octmnist':
            img = self.images[self.indices[idx]]
            img = np.stack((img,)*3, axis=-1)  #Converting to RGB

        if len(self.aug_list) !=0:
            nda = np.random.choice(self.aug_list, 1)[0]
            if nda == 'augmix':
                img = aug(Image.fromarray(img), self.transform, args.augmix_severity)

            elif nda == 'jigsaw':
                img = self.transform(Image.fromarray(img))
                img = jigsaw_k(img.unsqueeze(0))
                img = img.squeeze(0)

            elif nda == 'augmix_jigsaw':
                img = aug(Image.fromarray(img), self.transform, args.augmix_severity)
                img = jigsaw_k(img.unsqueeze(0))
                img = img.squeeze(0)

        else:
            img = self.transform(Image.fromarray(img))

        label = np.array([0])  #Dummy label
        label = torch.from_numpy(label)

        if self.use_cache:
            self.cache[idx] = (img, label)

        return img, label

    def __len__(self):
        return len(self.labels)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training with Calibration Protocol: Latent Space Inliers and Pixel Space Outliers')
    parser.add_argument('--in_dataset', default='bloodmnist', type=str, help='Name of the Dataset (bloodmnist|dermamnist ...)')
    parser.add_argument('--model_type', default='wrn', type=str, help='Model type')
    parser.add_argument('--dist', action='store_false', help='Distributed Training')
    parser.add_argument('--T', default=1.0, type=float, help='Temperature for the Energy function')
    parser.add_argument('--m_in', default=-20.0, type=float, help='Margin for In-dataset')
    parser.add_argument('--m_out', default=-7.0, type=float, help='Margin for Out-dataset')
    parser.add_argument('--beta1', default=0.1, type=float, help='Weight for margin based loss')
    parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)')

    parser.add_argument('--cutmix', action='store_true', help='Cut-mix Training')
    parser.add_argument('--jigsaw', action='store_true', help='Jigsaw required or not')
    parser.add_argument('--augmix', action='store_true', help='Augmix required or not')
    parser.add_argument('--augmix_severity', default=11, type=int, help='Outlier augmix severity')
    parser.add_argument('--rand_conv', action='store_true', help='Rand Conv required or not')
    parser.add_argument('--rand_conv_kernel_size', type=int, default=[9,11,13,15,17,19], nargs='+',help='kernel size for random layer, could be multiple kernels for multiscale mode')

    parser.add_argument('--start_epoch', type=int, default=2)
    parser.add_argument('--sample_number', type=int, default=250)
    parser.add_argument('--select', type=int, default=64)
    parser.add_argument('--sample_from', type=int, default=10000)
    parser.add_argument('--ckpt_name', default='in_latent_out_pix', type=str, help='Name of ckpt file')

    # WRN Architecture
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')

    args = parser.parse_args()
    return args


class TrainManager(object):
    """
    Class to manage model training
    :param model: Classifier model - VGG
    :param optimizer : Optimizer (Adam|SGD)
    :param params : Parameters - dict (In dataset hyper-parameters and settings)
    :param DataLoader train_loader: iterate through labeled in_train data
    :param DataLoader val_loader: iterate through in_val data
    :param DataLoader ood_loader: iterate through OOD data

    :return: object of TrainManager class
    """
    def __init__(self, model=None, optimizer=None, params={}, train_loader=None, val_loader=None, ood_loader=None):

        self.model = model
        self.params = params
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'],
                            betas=(0.9, 0.98), weight_decay=self.params['weight_decay'])

        # Loading previously stored checkpoint
        else:
            self.optimizer = optimizer

        self.device = self.params['device']
        self.in_train_loader = train_loader
        self.in_val_loader = val_loader
        self.ood_loader = ood_loader

        # TODO: add cosine annealing if required
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=self.params['lr_patience'],
                            factor=self.params['lr_factor'], verbose=True)

        # Dealing with Imbalanced data
        label_dist = np.unique(self.in_train_loader.dataset.labels, return_counts=True)[1]
        class_weights = [1 - (x / sum(label_dist)) for x in label_dist]
        class_weights = torch.FloatTensor(class_weights).to(self.params['device'])

        self.in_criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.softmax = nn.Softmax(dim=1)

        if args.rand_conv:
            print('Performing {}'.format('rand_conv'))
            data_mean = [0.485, 0.456, 0.406]
            data_std = [0.229, 0.224, 0.225]
            self.rand_module = get_random_module(self.model, data_mean, data_std, args.rand_conv_kernel_size, mixing=False)
            self.rand_module.to(self.device)

        self.checkpoint_name = args.ckpt_name+'_m_in_'+str(args.m_in)+'-m_out_'+str(args.m_out)+'-T_'+str(args.T)+'-ckpt.pth.tar'
        self.final_checkpoint_name = args.ckpt_name+'_m_in_'+str(args.m_in)+'-m_out_'+str(args.m_out)+'-T_'+str(args.T)+'-ckpt_last.pth.tar'


    def train(self):
        """
        Trains an OE model
        """
        best_val_acc = -1
        if args.model_type =='wrn':
            num_features = 128
        num_classes = self.params['num_classes']

        data_dict = torch.zeros(num_classes, args.sample_number, num_features).cuda()
        number_dict = {}
        for i in range(num_classes):
            number_dict[i] = 0
        eye_matrix = torch.eye(num_features).to(self.params['device'])

        for epoch in range(self.params['checkpoint_epoch'], self.params['epochs'] + 1):
            # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
            self.ood_loader.dataset.offset = np.random.randint(len(self.ood_loader.dataset))
            ood_iter = self.ood_loader.__iter__()

            # Set the model to train
            self.model.train()

            # Variables / Lists to store meta-information
            avg_loss = 0.
            total_probs = []
            total_true = []
            batch_time = 0.
            end = time.time()

            # Scheduler step
            self.lr_scheduler.step(epoch)

            # Iterating through the in_train_loader
            for i, in_set in enumerate(self.in_train_loader):

                # since len(ood_loader) < len(train_loader), reset iterator when you reach the end
                try:
                    out_set = next(ood_iter)
                except StopIteration:
                    del ood_iter
                    ood_iter = self.ood_loader.__iter__()
                    out_set = next(ood_iter)

                if args.rand_conv:
                     # Performing random conv
                     prob = np.random.random_sample()
                     if prob > 0.5:
                         self.rand_module.randomize()
                         rc = self.rand_module(in_set[0].clone().to(self.params['device']))
                         #print(rc.shape)
                         out_set[0] = torch.cat((out_set[0].to(self.params['device']), rc[np.random.choice(range(rc.shape[0]), 32)]), 0)



                # Concatenating in and out data along the batch axis
                input = torch.cat((in_set[0].to(self.params['device']), out_set[0].to(self.params['device'])), 0)
                input = input.to(self.params['device'])
                nat_input = input.clone()  # nat is native without corruption (No corruptions/adversarial perturbations)

                # Target available only from the in distribution data
                target = in_set[1]
                target = target.to(self.params['device'])

                # Input --> Model --> Output
                nat_output, features = self.model(nat_input)   # nat_output is the set of predicted logits
                nat_probs = self.softmax(nat_output)    # Softmax probabilites of [in, out] data samples

                # Separating the logits / softmax probs for the in and out samples
                nat_in_output = nat_output[:len(in_set[0])]  # In logits
                nat_out_output = nat_output[len(in_set[0]):] # Out logits
                nat_in_probs = nat_probs[:len(in_set[0])] # In softmax probs
                nat_out_probs = nat_probs[len(in_set[0]):] # Out softmax probs

                output = features[:len(in_set[0])]

                # Energy based regularization.
                sum_temp = 0
                for index in range(num_classes):
                    sum_temp += number_dict[index]
                nat_margin_loss = torch.zeros(1).to(self.params['device'])[0]

                if sum_temp == num_classes * args.sample_number and epoch < args.start_epoch:
                    # maintaining an ID data queue for each class.
                    target_numpy = target.cpu().data.numpy()
                    for index in range(len(target)):
                        dict_key = target_numpy[index]
                        data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                              output[index].detach().view(1, -1)), 0)

                elif sum_temp == num_classes * args.sample_number and epoch >= args.start_epoch:
                    target_numpy = target.cpu().data.numpy()
                    for index in range(len(target)):
                        dict_key = target_numpy[index]
                        data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                              output[index].detach().view(1, -1)), 0)
                    # the covariance finder needs the data to be centered.
                    for index in range(num_classes):
                        if index == 0:
                            X = data_dict[index] - data_dict[index].mean(0)
                            mean_embed_id = data_dict[index].mean(0).view(1, -1)
                        else:
                            X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
                            mean_embed_id = torch.cat((mean_embed_id,
                                                       data_dict[index].mean(0).view(1, -1)), 0)

                    ## add the variance.
                    temp_precision = torch.mm(X.t(), X) / len(X)
                    temp_precision += 0.001 * eye_matrix  #0.0001


                    for index in range(num_classes):
                        try:
                            new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                                mean_embed_id[index], covariance_matrix=temp_precision)
                            negative_samples = new_dis.rsample((args.sample_from,))
                            prob_density = new_dis.log_prob(negative_samples)
                            # breakpoint()
                            # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                            # keep the data in the low density area.
                            cur_samples, index_prob = torch.topk(- prob_density, args.select)
                            if index == 0:
                                outlier_samples = negative_samples[index_prob]
                            else:
                                outlier_samples = torch.cat((outlier_samples, negative_samples[index_prob]), 0)
                        except:
                            outlier_samples = []
                            pass
                    if len(outlier_samples) != 0:

                        if args.dist:
                            predictions_outlier_samples = self.model.module.fc(outlier_samples)
                        else:
                            predictions_outlier_samples = self.model.fc(outlier_samples)
                        energy_score_for_fg = -args.T * torch.logsumexp(predictions_outlier_samples/args.T, dim=1)
                        energy_score_for_bg = -args.T * torch.logsumexp(nat_out_output/args.T, dim=1)
                        # Minimize the energy for the latent inliers and maximize the energy for the latent space outliers
                        nat_margin_loss = torch.pow(F.relu(energy_score_for_fg - args.m_in), 2).mean() + torch.pow(F.relu(args.m_out-energy_score_for_bg), 2).mean()

                else:
                    target_numpy = target.cpu().data.numpy()
                    for index in range(len(target)):
                        dict_key = target_numpy[index]
                        if number_dict[dict_key] < args.sample_number:
                            data_dict[dict_key][number_dict[dict_key]] = output[index].detach()
                            number_dict[dict_key] += 1

                # Loss definitions
                nat_in_loss = self.in_criterion(nat_in_output, target)  # Cross-entropy for in_data
                # Compute gradient and do backprop
                loss = nat_in_loss
                loss += args.beta1 * nat_margin_loss #+ 0.01 * nat_uniform_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Compute training statistics
                pred = (np.argmax(nat_in_probs.data.cpu().numpy(),1)).astype(float)
                num_correct = (pred == target.data.cpu().numpy()).sum()
                batch_nat_acc = balanced_accuracy_score(target.data.cpu().numpy(), pred)
                # measure elapsed time
                batch_time = time.time() - end
                end = time.time()

                if i % args.print_freq == 0:
                    print('Epoch: [{}][{}/{}]\t'
                            'Time {:.3f}\t'
                            'Loss In {:.4f}\t'
                            'Loss Margin {:.4f}\t'
                            'Balanced Batch Acc {:.3f}'.format(epoch, i, len(self.in_train_loader), batch_time, nat_in_loss.item(), nat_margin_loss.item(), batch_nat_acc))

                total_probs.extend(nat_in_probs.data.cpu().numpy())
                total_true.extend(target.data.cpu().numpy())
                avg_loss += loss.item()/len(self.in_train_loader)

            print("Epoch {}/{}".format(epoch, self.params['epochs']))

            if 'mnist' in args.in_dataset:
                train_auc = roc_auc_score(np.array(total_true), np.array(total_probs), multi_class='ovo', average='weighted')

            train_acc = balanced_accuracy_score(np.array(total_true), np.argmax(np.array(total_probs),1))
            print("Train Loss {:.3f}\t Train AUC {:.3f}\t Train Balanced Acc {:.3f}".format(avg_loss, train_auc, train_acc))

            # Evaluate on validation set
            val_acc, val_loss = self.validate(epoch)

            #Model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print('Saving model at Epoch {}'.format(epoch))
                # remember best val_loss and save checkpoint
                if args.dist and torch.cuda.device_count() > 1:
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'optimizer': self.optimizer.state_dict(),
                        'state_dict': self.model.module.state_dict(),
                    }, epoch + 1, self.checkpoint_name)
                else:
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'optimizer': self.optimizer.state_dict(),
                        'state_dict': self.model.state_dict(),
                        }, epoch + 1, self.checkpoint_name)

        # save model for last epoch
        if args.dist and torch.cuda.device_count() > 1:
            self.save_checkpoint({
                'epoch': epoch + 1,
                'optimizer': self.optimizer.state_dict(),
                'state_dict': self.model.module.state_dict(),
            }, epoch + 1, self.final_checkpoint_name)
        else:
            self.save_checkpoint({
                'epoch': epoch + 1,
                'optimizer': self.optimizer.state_dict(),
                'state_dict': self.model.state_dict(),
                }, epoch + 1, self.final_checkpoint_name)


    def validate(self, epoch):
        """
        Validates a pytorch nn model
        :return: Balanced accuracy and AUC of best model on validation data
        """

        self.model.eval()
        with torch.no_grad():
            batch_time = 0.
            end = time.time()
            total_probs = []
            total_true = []
            avg_loss = 0.

            for i, (input, target) in enumerate(self.in_val_loader):
                input = input.to(self.params['device'])
                target = target.to(self.params['device'])

                # compute output
                output, _ = self.model(input)
                loss = self.in_criterion(output, target)

                # measure batch metrics
                probs = self.softmax(output)
                pred = (np.argmax(probs.data.cpu().numpy(),1)).astype(float)
                num_correct = (pred == target.data.cpu().numpy()).sum()
                batch_acc = num_correct/input.size(0)
                total_probs.extend(probs.data.cpu().numpy())
                total_true.extend(target.data.cpu().numpy())

                # measure elapsed time
                batch_time = time.time() - end
                end = time.time()

                if i % args.print_freq == 0:
                    print('Validation: [{}/{}]\t'
                          'Time {:.3f}\t'
                          'Loss {:.4f}\t'
                          'Acc {:.3f}'.format(i, len(self.in_val_loader), batch_time, loss.item(), batch_acc))

                avg_loss += loss.item()/len(self.in_val_loader)

            total_true = np.array(total_true)
            total_probs = np.array(total_probs)
            if 'mnist' in args.in_dataset:
                val_auc = roc_auc_score(total_true, total_probs, multi_class='ovo', average='weighted')

            val_acc = balanced_accuracy_score(total_true, np.argmax(total_probs,1))
            print('Auc {:.3f} \t''Balanced Acc {:.3f}'.format(val_auc, val_acc))
            print("Confusion Matrix:")
            print(confusion_matrix(total_true, np.argmax(total_probs,1)))

            return val_acc, avg_loss


    def save_checkpoint(self, state, epoch, name):
        """Saves checkpoint to disk"""
        directory = os.path.join(self.params['checkpoint_dir'],args.model_type, args.in_dataset, args.ckpt_name)
        filename = directory + '/' + name
        torch.save(state, filename)

################################################################################
def main():

    # Get the hyper-parameters as well as the settings for 'in_dataset' & 'out_dataset'
    in_dataset_params = hparams_registry.default_hparams(args.in_dataset)

    if args.model_type == 'wrn':
        in_dataset_params['im_size'] = 32

    ############################################################################

    # CUDA / CPU
    if torch.cuda.is_available():
        in_dataset_params['device'] = 'cuda'
        print('Experiment Running on cuda')
    else:
        in_dataset_params['device'] = 'cpu'
        print('Experiment Running on cpu')

    ############################################################################

    # Distributed training and Directory Creation
    if args.dist and torch.cuda.is_available():
        print('Distributed Training')
        torch.distributed.init_process_group("nccl", init_method="env://",
            world_size=int(os.environ['SLURM_NPROCS']),
            rank=int(os.environ['SLURM_PROCID']))

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        ngpus_per_node = torch.cuda.device_count()
        local_rank = rank % ngpus_per_node
        in_dataset_params['device'] = torch.device(f"cuda:{local_rank}") #torch.cuda.device(local_rank)
        ood_dataset_params['device'] = torch.device(f"cuda:{local_rank}")

        torch.cuda.set_device(local_rank)

        if rank == 0:
            if not os.path.exists(os.path.join(in_dataset_params['checkpoint_dir'],args.model_type, args.in_dataset, args.ckpt_name)):
                os.makedirs(os.path.join(in_dataset_params['checkpoint_dir'],args.model_type, args.in_dataset, args.ckpt_name))
            if not os.path.exists(os.path.join(in_dataset_params['result_dir'],args.model_type, args.in_dataset, args.ckpt_name)):
                os.makedirs(os.path.join(in_dataset_params['result_dir'],args.model_type, args.in_dataset, args.ckpt_name))

    # Regular training
    else:
        if not os.path.exists(os.path.join(in_dataset_params['checkpoint_dir'],args.model_type, args.in_dataset, args.ckpt_name)):
            os.makedirs(os.path.join(in_dataset_params['checkpoint_dir'],args.model_type, args.in_dataset, args.ckpt_name))
        if not os.path.exists(os.path.join(in_dataset_params['result_dir'],args.model_type, args.in_dataset, args.ckpt_name)):
            os.makedirs(os.path.join(in_dataset_params['result_dir'],args.model_type, args.in_dataset, args.ckpt_name))

    ############################################################################

    # Train and Val Loader for 'in_dataset'
    print("Loading Dataloaders")
    in_train_loader, in_val_loader = get_loaders(args.in_dataset, in_dataset_params, args.dist)

    ############################################################################
    # Loader for 'out_dataset'
    # Dataset with options for jigsaw, augmix and rand conv individually
    # Dataset with Imagenet samples
    print('Augmentations Chosen')
    print('Augmix - {}'.format(args.augmix))
    print('Jigsaw - {}'.format(args.jigsaw))
    print('Rand Conv - {}'.format(args.rand_conv))

    aug_list = []
    if args.augmix:
        aug_list.append('augmix_jigsaw')
    if args.jigsaw:
        aug_list.append('jigsaw')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tt = [transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05)),transforms.Resize((in_dataset_params['im_size'],in_dataset_params['im_size']))]
    tt.extend([transforms.ToTensor(), normalize])
    trans = transforms.Compose(tt)

    neg_aug_dataset = In_Dist_Neg_Aug(args.in_dataset, in_dataset_params['train_csv'], in_dataset_params, trans, in_dataset_params['device'], aug_list, use_cache=False)


    if args.dist:
        sampler = get_weighted_sampler(np.zeros(len(neg_aug_dataset.labels)))
        sampler = DistributedSamplerWrapper(sampler)
        print('Obtained Distributed Sampler')
        ood_loader = DataLoader(dataset=neg_aug_dataset, batch_size=in_dataset_params['batch_size'], sampler=sampler,
                              num_workers=in_dataset_params['num_workers']//2)

    else:
        ood_loader = DataLoader(dataset=neg_aug_dataset, batch_size=in_dataset_params['batch_size'], shuffle=True,
                              num_workers=in_dataset_params['num_workers'])

    ############################################################################
    if args.model_type == 'wrn':
        from modeldefs.mnist_wrn import MNISTWideResNet
        model = MNISTWideResNet(args.layers, in_dataset_params['num_classes'], args.widen_factor, dropRate=args.droprate)
        print('Chosen model = {}'.format(args.model_type))
        model = model.to(in_dataset_params['device'])
        print('Loaded model')
        print(list(model.children()))
    else:
        raise NotImplementedError

    ############################################################################

    # Checkpoint File / Resuming Training
    checkpoint_file = os.path.join(in_dataset_params['checkpoint_dir'],args.model_type, args.in_dataset, args.ckpt_name, args.ckpt_name+f'_m_in_{args.m_in}-m_out_{args.m_out}-T_{args.T}-ckpt.pth.tar')
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['state_dict'])
        in_dataset_params['checkpoint_epoch'] = checkpoint['epoch']
        print(checkpoint['epoch'])

        optimizer = torch.optim.Adam(model.parameters(), lr=in_dataset_params['lr'],
                                     betas=(0.9, 0.98), weight_decay=in_dataset_params['weight_decay'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded checkpoint")
    else:
        in_dataset_params['checkpoint_epoch'] = 1
        optimizer = None
        print("No existing checkpoint")

    ############################################################################

    # Multi-gpu training if available
    if args.dist and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        model.to(local_rank)
    else:
        model = model.to(in_dataset_params['device'])

    # Get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))


    ############################################################################
    print('In dataset params')
    print(in_dataset_params)

    manager = TrainManager(model, optimizer, in_dataset_params, in_train_loader, in_val_loader, ood_loader)
    manager.train()

# Code starts here
if __name__ == '__main__':
    # Storing the starting time
    start = time.time()

    # Parsing the arguments
    args = parse_arguments()
    print(args)

    main()

    # Program completion verbose
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
