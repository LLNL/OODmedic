import torch
#from models import VGG, VGG16Net, LossEstimator
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from PIL import Image
import cv2
import os
import sys
sys.path.append(os.path.abspath('../../'))

'''
tmp_model = VGG.from_pretrained('vgg16', num_classes=5)
#print(list(tmp_model.children()))
m = VGG16Net(tmp_model)
l = LossEstimator()
x = torch.randn(3,3,224,224)
y_logits, features = m(x)
print(y_logits)
print([features[i].shape for i in range(len(features))])
loss = l(features)
print(loss.shape)
'''
tt = []
imsize = 224
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
tt.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
#tt.extend([transforms.Resize((imsize,imsize))])
tt.extend([transforms.CenterCrop(imsize)])
tt.extend([transforms.ToTensor(), normalize])

trans = transforms.Compose(tt)

datadir = '../../data/ISIC_2019_Training_Input'
imgpath = 'ISIC_0069267'
img = Image.open(os.path.join(datadir, imgpath+'.jpg'))
img = trans(img)
plt.imshow(img.permute(1, 2, 0).data.numpy())
plt.savefig('test.png')
#print(img.shape)
