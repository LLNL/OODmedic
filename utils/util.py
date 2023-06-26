import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
import time
from torch.autograd import Variable

def contrastive_loss(input, target, margin=1.0, reduction='mean'):
    if len(input) % 2 == 1:
        input = input[:(len(input)-1)]
        target = target[:(len(target)-1)]
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()
    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)
    else:
        loss = torch.clamp(margin - one * input, min=0)
    return loss



def sample_estimator(self):

    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance
    with torch.no_grad():

        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

        temp_x = torch.rand(2,3,self.params['im_size'],self.params['im_size']).to(self.params['device'])
        _, temp_list = self.model.module.feature_list(temp_x)
        num_output = len(temp_list)
        feature_list = np.empty(num_output)
        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)   # No. of feature maps
            count += 1

        num_output = len(feature_list)
        num_sample_per_class = np.empty(self.params['num_classes'])
        num_sample_per_class.fill(0)
        list_features = []

        for i in range(num_output):
            temp_list = []
            for j in range(self.params['num_classes']):
                temp_list.append(0)
            list_features.append(temp_list)

        print('Computing Mean for every layer for every class')
        t = time.time()

        for i, (data, target) in enumerate(self.train_loader):
            print('Processing Batch {}/{}'.format(i+1, len(self.train_loader)))
            data = data.to(self.params['device'])
            output, out_features = self.model.module.feature_list(data)

            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)


            # construct the sample matrix
            for i in range(data.size(0)):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = out[i].view(1, -1)
                        out_count += 1
                else:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] \
                        = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                        out_count += 1
                num_sample_per_class[label] += 1

        sample_class_mean = []
        out_count = 0
        for num_feature in feature_list:
            temp_list = torch.Tensor(self.params['num_classes'], int(num_feature)).to(self.params['device'])
            for j in range(self.params['num_classes']):
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
            sample_class_mean.append(temp_list)
            print('M', temp_list.shape)
            out_count += 1

        print('Computed the means : Time taken {}'.format(round(time.time()-t, 2)))
        print('Computing the precision matrix')
        precision = []
        for k in range(num_output):
            X = 0
            for i in range(self.params['num_classes']):
                if i == 0:
                    X = list_features[k][i] - sample_class_mean[k][i]
                else:
                    X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
            # find inverse
            group_lasso.fit(X.cpu().data.numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float().to(self.params['device'])
            print('Cov', temp_precision.shape)
            precision.append(temp_precision)


    directory = os.path.join(self.params['result_dir'],self.args.model_type, self.args.in_dataset, self.args.dir_name, self.args.algo_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename_mean = directory + '/mean.pkl'
    filename_precision = directory + '/precision.pkl'
    with open(filename_mean, 'wb') as f:
        pickle.dump(sample_class_mean, f)

    with open(filename_precision, 'wb') as f:
        pickle.dump(precision, f)

    print('Computed the precision matrix : Time taken {}'.format(round(time.time()-t, 2)))
    return sample_class_mean, precision


def get_Mahalanobis_score(self, data, sample_mean, precision, magnitude):

    num_output = 4 #len(features)
    for layer_index in range(num_output):
        data = Variable(data, requires_grad = True)
        _, out_features = self.model.module.feature_list(data)
        out_features = out_features[layer_index]
        #print(out_features.shape)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        #print(out_features.shape)

        gaussian_score = 0
        for i in range(self.params['num_classes']):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient[:,0] = (gradient[:,0] )/(0.229)
        gradient[:,1] = (gradient[:,1] )/(0.224)
        gradient[:,2] = (gradient[:,2])/(0.225)

        tempInputs = torch.add(data.data, -magnitude, gradient)

        _, noise_out_features = self.model.module.feature_list(Variable(tempInputs))
        noise_out_features = noise_out_features[layer_index]
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(self.params['num_classes']):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)

        noise_gaussian_score = np.asarray(noise_gaussian_score.cpu().numpy(), dtype=np.float32)
        if layer_index == 0:
            Mahalanobis_scores = noise_gaussian_score.reshape((noise_gaussian_score.shape[0], -1))
        else:
            Mahalanobis_scores = np.concatenate((Mahalanobis_scores, noise_gaussian_score.reshape((noise_gaussian_score.shape[0], -1))), axis=1)

    #print(Mahalanobis_scores.shape)
    return Mahalanobis_scores

def get_perm(l) :
    perm = torch.randperm(l)
    while torch.all(torch.eq(perm, torch.arange(l))) :
        perm = torch.randperm(l)
    return perm

def cutmix(input, beta, device):
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(input.size()[0]).to(device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    return input

# Originally k=2
def jigsaw_k(data, k = 4) :
    with torch.no_grad() :
        actual_h = data.size()[2]
        actual_w = data.size()[3]
        h = torch.split(data, int(actual_h/k), dim = 2)
        splits = []
        for i in range(k) :
            splits += torch.split(h[i], int(actual_w/k), dim = 3)
        fake_samples = torch.stack(splits, -1)
        for idx in range(fake_samples.size()[0]) :
            perm = get_perm(k*k)
            # fake_samples[idx] = fake_samples[idx,:,:,:,torch.randperm(k*k)]
            fake_samples[idx] = fake_samples[idx,:,:,:,perm]
        fake_samples = torch.split(fake_samples, 1, dim=4)
        merged = []
        for i in range(k) :
            merged += [torch.cat(fake_samples[i*k:(i+1)*k], 2)]
        fake_samples = torch.squeeze(torch.cat(merged, 3), -1)
        return fake_samples

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
