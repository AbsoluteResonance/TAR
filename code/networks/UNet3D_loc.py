# -*- coding: utf-8 -*-
"""
An implementation of the 3D U-Net paper:
     Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger:
     3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. 
     MICCAI (2) 2016: 424-432
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
The implementation is borrowed from: https://github.com/ozan-oktay/Attention-Gated-Networks
"""
import sys
# sys.path.append('..')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks_other import init_weights
from .utils import UnetConv3, UnetUp3, UnetUp3_CT


class UNET_3D(nn.Module):

    def __init__(self, in_channels=3, feature_scale=4, n_classes=21, known_n_points=None, height=128, width=128, depth=64, is_deconv=True, is_batchnorm=True):
        super(UNET_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters] #  [16, 32, 64, 128, 256] when `feature_scale=4`

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        
        # Add one more downsampling considering `feature_scale=5`
        # self.maxpool5 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        # self.conv5 = UnetConv3(filters[4], 512, is_batchnorm=False, kernel_size=(
        #     3, 3, 3), padding_size=(1, 1, 1))

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)
        self.tanh = nn.Tanh()

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        # Initialize params
        self.sigmoid = nn.Sigmoid()
        self.known_n_points = known_n_points
        if known_n_points is None:
            steps = 4 # four upsampling blocks
            mid_h = height // (2**steps)
            mid_w = width // (2**steps)
            mid_d = depth // (2**steps)
            # print(f"h: {mid_h} w: {mid_w} d: {mid_d}")

            # Transform the Bottleneck feature maps
            self.btlnk_branch = nn.Sequential(
                nn.Linear(mid_h*mid_w*mid_d*256, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5)
            )

            # Transform and vectorize the decoded features
            self.featmap_branch = nn.Sequential(
                nn.Linear(height*width*depth, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5)
            )

            # Regression Layer for Lesion location prediction
            self.regressor = nn.Sequential(
                nn.Linear(64+64, 1),
                nn.ReLU()
            )

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)      # torch.Size([4, 16, 64, 128, 128])
        maxpool1 = self.maxpool1(conv1) # torch.Size([4, 16, 32, 64, 64])

        conv2 = self.conv2(maxpool1)    # torch.Size([4, 32, 32, 64, 64])
        maxpool2 = self.maxpool2(conv2) # torch.Size([4, 32, 16, 32, 32])

        conv3 = self.conv3(maxpool2)    # torch.Size([4, 64, 16, 32, 32])
        maxpool3 = self.maxpool3(conv3) # torch.Size([4, 64, 8, 16, 16])

        conv4 = self.conv4(maxpool3)    # torch.Size([4, 128, 8, 16, 16])
        maxpool4 = self.maxpool4(conv4) # torch.Size([4, 128, 4, 8, 8])
        
        # Bottleneck
        # maxpool5 = self.maxpool5(center)
        # conv5 = self.conv5(maxpool5)
        # print(f'conv5: {conv5.size()}')
        center = self.center(maxpool4)
        center = self.dropout1(center)          # torch.Size([4, 256, 4, 8, 8]) is the deeper feature for the fusion with the final
        up4 = self.up_concat4(conv4, center)    # torch.Size([4, 128, 8, 16, 16])
        up3 = self.up_concat3(conv3, up4)       # torch.Size([4, 64, 16, 32, 32])
        up2 = self.up_concat2(conv2, up3)       # torch.Size([4, 32, 32, 64, 64])
        up1 = self.up_concat1(conv1, up2)       # torch.Size([4, 16, 64, 128, 128])
        up1 = self.dropout2(up1)

        final = self.final(up1)                 # torch.Size([4, 1, 64, 128, 128])

        bs = inputs.size(0)
        if self.known_n_points is None:
            bottleneck_layer = center
            bottleneck_layer_flat = bottleneck_layer.view(bs, -1)   # torch.Size([4, 256*4*8*8]) 
            pred_feature_flat = self.sigmoid(final).view(bs, -1)                  # torch.Size([4, 1*64*128*128])
            
            lateral_feature = self.btlnk_branch(bottleneck_layer_flat) # torch.Size([4, 64])
            pred_feature_flat = self.featmap_branch(pred_feature_flat) # torch.Size([4, 64])

            # Concat both features for regression
            regression_features = torch.cat([lateral_feature, pred_feature_flat], dim=1) # torch.Size([4, 128])

            regression = self.regressor(regression_features) # torch.Size([4, 1])

        return final, self.sigmoid(final), regression

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
    
if __name__=="__main__":
    import torch
    x = torch.randn((4, 1, 64, 96, 96)).cuda()
    model = UNET_3D(in_channels=1, feature_scale=4, n_classes=1, height=96, width=96, depth=64).cuda()
    out, regresion = model(x)
    print(f"out: {out.shape}") # out: torch.Size([4, 1, 64, 112, 112])
    print(f"regresion: {regresion.shape}")
    # print(f'center-layer: {model}')
    print(f"Total Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    """
    out: torch.Size([4, 1, 64, 96, 96])
    regresion: torch.Size([4, 1])
    Total Params: 45.99M
    """