import torch
from torch import nn
from resnet import resnet50


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x

class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class C4_head(nn.Module):
    def __init__(self,in_channel=256,out_channel=512):
        super(C4_head, self).__init__()

        self.conv1 = nn.Conv3d(in_channel,out_channel, kernel_size=(3,3,3), stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 2), stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(out_channel, out_channel*2, kernel_size=(2,2,1), stride=1, padding=0, bias=False)

    def forward(self, x, bs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        bs_num,c,w,h,d = x.shape
        x= torch.reshape(x,(bs,bs_num//bs,c*w*h*d))
        return x

class C5_head(nn.Module):
    def __init__(self,in_channel=512,out_channel=1024):
        super(C5_head, self).__init__()

        self.conv1 = nn.Conv3d(in_channel,out_channel, kernel_size=(3,3,2), stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, bs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        bs_num, c, w, h, d = x.shape
        x = torch.reshape(x, (bs, bs_num // bs, c * w * h * d))
        return x

class Resnet50(nn.Module):
    def __init__(self, resnet_encoder=None, n_channels=3, n_classes=2, n_filters=64, normalization='none', has_dropout=False):
        super(Resnet50, self).__init__()
        self.has_dropout = has_dropout
        self.resnet_encoder = resnet50(in_channel=n_channels)

        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        # Add tanh activation
        self.out_conv2 = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.tanh = nn.Tanh()

        self.__init_weight()


    def decoder(self, features):
        x1 = features[0] # torch.Size([4, 16, 112, 112, 64])
        x2 = features[1] # torch.Size([4, 32, 56, 56, 32])
        x3 = features[2] # torch.Size([4, 64, 28, 28, 16])
        x4 = features[3] # torch.Size([4, 128, 14, 14, 8])
        x5 = features[4] # torch.Size([4, 256, 7, 7, 4])
        
        x5_up = self.block_five_up(x5) # x5_up: torch.Size([4, 128, 14, 14, 8])
        x5_up = x5_up + x4 # torch.Size([4, 128, 14, 14, 8])

        x6 = self.block_six(x5_up) # torch.Size([4, 128, 14, 14, 8])
        x6_up = self.block_six_up(x6) # torch.Size([4, 64, 28, 28, 16])
        x6_up = x6_up + x3 # torch.Size([4, 64, 28, 28, 16])

        x7 = self.block_seven(x6_up) # torch.Size([4, 64, 28, 28, 16])
        x7_up = self.block_seven_up(x7) # torch.Size([4, 32, 56, 56, 32])
        x7_up = x7_up + x2 # torch.Size([4, 32, 56, 56, 32])

        x8 = self.block_eight(x7_up) # torch.Size([4, 32, 56, 56, 32])
        x8_up = self.block_eight_up(x8) # torch.Size([4, 16, 112, 112, 64])
        x8_up = x8_up + x1 # torch.Size([4, 16, 112, 112, 64])
        x9 = self.block_nine(x8_up) # torch.Size([4, 16, 112, 112, 64])
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9) # torch.Size([4, 1, 112, 112, 64])

        return out, x9

    def forward(self, input, batch_size=4):
        resnet_features = self.resnet_encoder(input)
        # resnet_output: [torch.Size([4, 16, 64, 128, 128]), 
        #                 torch.Size([4, 128, 32, 64, 64]), 
        #                 torch.Size([4, 256, 16, 32, 32]), 
        #                 torch.Size([4, 512, 8, 16, 16]),
        #                 torch.Size([4, 1024, 4, 8, 8])]
        print(f'resnet_output: {[f.shape for f in resnet_features]}')
        out, x9 = self.decoder(resnet_features)
        out_tanh = self.tanh(out)
        out_seg = self.out_conv2(x9)
        return out_tanh, out_seg
        # return out

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m,nn.ConvTranspose3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__=="__main__":
    x = torch.randn((4, 3, 64, 128, 128)).cuda()
    model = Resnet50(n_channels=3).cuda()
    out = model(x)  
    print(f'out: {out[0].shape}, out[1]: {out[1].shape}, out[2]: {out[2].shape}, out[3]: {out[3].shape}')