import torch
import torch.nn as nn
import torch.nn.functional as F

class _ASPPModule3D(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule3D, self).__init__()
        self.atrous_conv = nn.Conv3d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP3D(nn.Module):
    def __init__(self, inplanes=64, outplanes=64, output_stride=16, BatchNorm=nn.BatchNorm3d):
        super(ASPP3D, self).__init__()

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule3D(inplanes, outplanes, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule3D(inplanes, outplanes, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule3D(inplanes, outplanes, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule3D(inplanes, outplanes, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(inplanes, outplanes, 1, stride=1, bias=False)
        )
        self.bn_after_pool = BatchNorm(outplanes)
        self.relu_after_pool = nn.ReLU()

        self.conv1 = nn.Conv3d(outplanes * 5, outplanes, 1, bias=False)
        self.bn1 = BatchNorm(outplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        if x5.shape[0] > 1:
            x5 = self.bn_after_pool(x5)
        x5 = self.relu_after_pool(x5)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='trilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_aspp3d(inplanes=64, outplanes=64, output_stride=16, BatchNorm=nn.BatchNorm3d):
    return ASPP3D(inplanes, outplanes, output_stride, BatchNorm)

# Example usage
if __name__ == "__main__":
    aspp3d = build_aspp3d(inplanes=64, outplanes=64, output_stride=16, BatchNorm=nn.BatchNorm3d)
    input_tensor = torch.randn(2, 64, 16, 32, 32) 
    output_tensor = aspp3d(input_tensor)
    print("Output tensor shape:", output_tensor.shape)
