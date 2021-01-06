"""
Encoder for few shot segmentation (VGG16)
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np

class Pyramid_Module(nn.Module):

    def __init__(self, ):
        super(Pyramid_Module, self).__init__()
        self.avg_pooling = nn.AvgPool2d((3, 3), stride=(2, 2), padding = (1, 1))
        self.p1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=1,  padding=1),   
            nn.ReLU(inplace=True)
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=1,  padding=1),   
            nn.ReLU(inplace=True)
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=1,  padding=1),   
            nn.ReLU(inplace=True)
        )
        self.feat_fuse = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=1,  padding=1),   
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=1,  padding=1),   
            nn.ReLU(inplace=True)
        )

        self.residule1 = nn.Sequential(
            nn.Conv2d(512 , 64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.residule2 = nn.Sequential(
            nn.Conv2d(512 , 64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.relu = nn.ReLU(inplace=True)

        
        
    def forward(self, feature):

        
        feat_1 = self.p1(feature)
        
        feature_2 = self.avg_pooling(feature)
        feat_2 = self.p2(feature_2)
        feat_2 = F.interpolate(feat_2, size= feature.shape[-2:], mode='bilinear')
        
        feature_3 = self.avg_pooling(feature_2)
        feat_3 = self.p3(feature_3)
        feat_3 = F.interpolate(feat_3, size= feature.shape[-2:], mode='bilinear')

        out = feat_1 + feat_2 + feat_3
        out = self.feat_fuse(out)

        out = out + self.residule1(out)
        out = self.relu(out)
        out = out + self.residule2(out)
        out = self.relu(out)


        return out


class Classifier_Module(nn.Module):

    def __init__(self, dims_in, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(dims_in, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class Encoder(nn.Module):
    """
    Encoder for few shot segmentation

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
    """
    def __init__(self,  pretrained_path=None):
        super().__init__()
        vgg = models.vgg16()
        
        #vgg.load_state_dict(torch.load(pretrained_path))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        #remove pool4/pool5
        features = nn.Sequential(*(features[i] for i in list(range(23))+list(range(24,30))))
        
        for i in [23,25,27]:
            features[i].dilation = (2,2)
            features[i].padding = (2,2)
        
            
        fc6 = nn.Conv2d(512, 512, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(512, 512, kernel_size=3, padding=4, dilation=4)

        self.features = nn.Sequential(*([features[i] for i in list(range(len(features)))] + [ fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))

        self.classifier = Classifier_Module(512, [6,12,18,24],[6,12,18,24],21) #only 15+1 classes is used for loss calculation

        self.feat_rd = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, dilation=1,  padding=1),   
            nn.ReLU(inplace=True)
        )

        self.pyramid_m = Pyramid_Module()

        self.classifier_2 = Classifier_Module(512, [6,12,18,24],[6,12,18,24],2)

    def forward(self, sup_imgs, qry_imgs, sup_mask):
        sup_feat = self.features(sup_imgs)

        sup_feat_interp = F.interpolate(sup_feat, size=sup_mask.shape[-2:], mode='bilinear')
        prototype = torch.sum(sup_feat_interp * sup_mask[None, ...], dim=(2, 3)) \
            / (sup_mask[None, ...].sum(dim=(2, 3)) + 1e-5) # N x C

        qry_feat = self.features(qry_imgs)
        n,c,h,w = qry_feat.size()
        prototype_q = prototype.view(n,c,1,1).expand(-1,-1,h,w)   #1*c*h*w
        feat_in = torch.cat([qry_feat,prototype_q], dim=1)
        feat_in = self.feat_rd(feat_in)
        feat_in = self.pyramid_m(feat_in)
        pred = self.classifier_2(feat_in)

        pred_temp = pred
        pred_temp = F.softmax(pred_temp, dim=1)
        pred_temp = F.interpolate(pred_temp, size= qry_imgs.shape[-2:], mode='bilinear').cpu().data[0].numpy()
        pred_temp = pred_temp.transpose(1,2,0)
        label, prob = np.argmax(pred_temp, axis=2), np.max(pred_temp, axis=2)
        
        label = torch.from_numpy(label).cuda().unsqueeze(0).unsqueeze(0)

        qry_feat_interp = F.interpolate(qry_feat, size=qry_imgs.shape[-2:], mode='bilinear')
        self_prototype = torch.sum(qry_feat_interp * label, dim=(2, 3)) \
            / (label.sum(dim=(2, 3)) + 1e-5) # N x C

        self_prototype = (self_prototype + prototype)/2.0

        self_prototype = self_prototype.view(n,c,1,1).expand(-1,-1,h,w)   #1*c*h*w
        feat_in = torch.cat([qry_feat,self_prototype], dim=1)
        feat_in = self.feat_rd(feat_in)
        feat_in = self.pyramid_m(feat_in)
        pred = self.classifier_2(feat_in)

        return pred

