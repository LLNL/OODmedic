import sys
import os
sys.path.append('./')
sys.path.append('../../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import torch
import torch.nn as nn
#from .utility import *
from torchvision import models
#from utility import get_model_params
#from utility import load_pretrained_weights
#from utility import vgg_params


configures = {
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    }
class VGG(nn.Module):

    def __init__(self, global_params=None):
        """ An VGGNet model. Most easily loaded with the .from_name or .from_pretrained methods
        Args:
          global_params (namedtuple): A set of GlobalParams shared between blocks
        Examples:
          model = VGG.from_pretrained('vgg11')
        """

        super(VGG, self).__init__()

        self.features = make_layers(configures[global_params.configure], global_params.batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            # nn.Dropout(global_params.dropout_rate),
            nn.Dropout(0.4),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            # nn.Dropout(global_params.dropout_rate),
            nn.Dropout(0.4),
            nn.Linear(4096, global_params.num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        print(x.shape, 'You can do this')
        print(list(self.features.children()))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        global_params = get_model_params(model_name, override_params)
        return cls(global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
        return model


    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (vgg{i} for i in 11,13,16,19) at the moment. """
        valid_models = ['vgg' + str(i) for i in ["11", "11_bn",
                                                 "13", "13_bn",
                                                 "16", "16_bn",
                                                 "19", "19_bn"]]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))


def make_layers(configure, batch_norm):
    layers = []
    in_channels = 3
    for v in configure:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
# D here corresponds to the filters as well as the Max pool operations needed to implement a VGG16
class VGG16Net(nn.Module):
    def __init__(self, original_model, conf=False):
        super(VGG16Net, self).__init__()
        self.features1 = nn.Sequential(*list(original_model.features.children())[0:4])
        self.features2 = nn.Sequential(*list(original_model.features.children())[4:9])
        self.features3 = nn.Sequential(*list(original_model.features.children())[9:16])
        self.features4 = nn.Sequential(*list(original_model.features.children())[16:23])
        self.features5 = nn.Sequential(*list(original_model.features.children())[23:30])
        self.final_max_pool = nn.Sequential(*list(original_model.features.children())[30:])
        self.avgpool = original_model.avgpool
        self.classifier = nn.Sequential(*list(original_model.classifier.children())[:6])

        self.classifier_final_layer = nn.Sequential(*list(original_model.classifier.children())[6:])

        self.conf = conf
        if self.conf:
            self.classifier_confidence_layer = torch.nn.Linear(4096, 1)

    def forward(self, x):

        x = self.features1(x)
        out1 = self.features2(x)
        out2 = self.features3(out1)
        out3 = self.features4(out2)
        out4 = self.features5(out3)
        x = self.final_max_pool(out4)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        pred_logits = self.classifier_final_layer(x)
        if self.conf:
            confidence = self.classifier_confidence_layer(x)
            return pred_logits, [out1, out2, out3, out4], confidence
        else:
            return pred_logits, [out1, out2, out3, out4]


class DUQ_VGG16Net(nn.Module): #output the W_c times output of the VGG16 net
    def __init__(self, original_model, features, num_embeddings):
        super(DUQ_VGG16Net, self).__init__()
        self.gamma = 0.99
        self.sigma = 0.3

        embedding_size = 10

        self.features1 = nn.Sequential(*list(original_model.features.children())[0:4])
        self.features2 = nn.Sequential(*list(original_model.features.children())[4:9])
        self.features3 = nn.Sequential(*list(original_model.features.children())[9:16])
        self.features4 = nn.Sequential(*list(original_model.features.children())[16:23])
        self.features5 = nn.Sequential(*list(original_model.features.children())[23:30])
        self.final_max_pool = nn.Sequential(*list(original_model.features.children())[30:])
        self.avgpool = original_model.avgpool
#         self.classifier = original_model.classifier
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            # nn.Dropout(global_params.dropout_rate),
            nn.Dropout(0.4),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            # nn.Dropout(global_params.dropout_rate),
            nn.Dropout(0.4),
            nn.Linear(4096, features), #outputs a feature space vector
        )

        self.W = nn.Parameter(torch.normal(torch.zeros(embedding_size, num_embeddings, features), 1)) # by 9 by ?
        # num_embedding is the number of classes : 9
        # features is the dim of the feature_space :
        # embedding_size is the dimension of the embedded space

        self.register_buffer('N', torch.ones(num_embeddings) * 20) #buffer is saved in the model state dicts but not trained.
        self.register_buffer('m', torch.normal(torch.zeros(embedding_size, num_embeddings), 1))

        self.m = self.m * self.N.unsqueeze(0) #initial m for eq 5 : normal distributed samples
        #initial N is ones
        #print(self.m)

    def embed(self, x):
        x = self.features1(x)
        out1 = self.features2(x)
        out2 = self.features3(out1)
        out3 = self.features4(out2)
        out4 = self.features5(out3)
        x = self.final_max_pool(out4)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        #print(x.type())
        #print(self.W.type())
        # i is batch, m is embedding_size, n is num_embeddings (classes), j is 20, dim of features.
        # embedding size(10=m) is the resulting dimension of the vector compared to the centroid.
        x = torch.einsum('ij,mnj->imn', x, self.W) #multiply and sum the axis corresponding to j

        return x

    def bilinear(self, z): #kernel used for the distance computation : criteria for OOD detection
        #z is batch, 10, 2
        embeddings = self.m / self.N.unsqueeze(0) #eq 6
#         print(self.m) # 10 by 2
#         print(self.N.unsqueeze(0))#[20,20]
#         print(embeddings) # 10 by 2

        diff = z - embeddings.unsqueeze(0)
        y_pred = (- diff**2).mean(1).div(2 * self.sigma**2).exp() #length = num_classes

        return y_pred

    def forward(self, x):
        z = self.embed(x) #features (imn : batch, 10, 2)
        y_pred = self.bilinear(z) #the distances

        return z, y_pred

    def update_embeddings(self, x, y):
        # x, y are the batches, so y is the class one hot encoded
        y = y.cuda()
        z = self.embed(x) #features
        # gamma is 0.99
        # normalizing value per class, assumes y is one_hot encoded : updATE SELF.N
        self.N = torch.max(self.gamma * self.N + (1 - self.gamma) * y.sum(0), torch.ones_like(self.N).cuda()) #eq 4

        # compute sum of embeddings on class by class basis
        features_sum = torch.einsum('ijk,ik->jk', z.float(), y.float())

        self.m = self.gamma * self.m + (1 - self.gamma) * features_sum #eq 5


class LossEstimator(nn.Module):
    def __init__(self, feature_sizes=[112, 56, 28, 14], num_channels=[128, 256, 512, 512], interm_dim=128):
        super(LossEstimator, self).__init__()

        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)

    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out



def main():
    ckpt_path = '/p/lustre1/viv41siv/medical-ood/vgg16-397923af.pth'
    # Model definition
    tmp_model = models.vgg16(pretrained=False)
    print(list(tmp_model.children()))
    tmp_model.load_state_dict(torch.load(ckpt_path))
    tmp_model.classifier[6] = torch.nn.Linear(4096, 8) # Changing the last layer of the VGG
    model = VGG16Net(tmp_model,conf=True) # VGG16Net will finally return pred logits and layers

    print('Loaded Model')
    print(list(model.children()))

    x = torch.randn(1,3,224,224)
    y, _, _ = model(x)
    print(x.shape, y.shape)

if __name__ == '__main__':
    main()
