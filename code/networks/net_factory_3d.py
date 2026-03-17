from .VNet import VNet
from .UNet3D_contrastive import UNet3D


def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2, scaler=4, use_aspp=False):
    if net_type == "unet_3D": 
        net = UNet3D(in_channels=in_chns, n_classes=class_num, scale_factor=scaler, use_aspp=use_aspp) # .cuda()
    elif net_type == "vnet": 
        net = VNet(n_channels=in_chns, n_classes=class_num, scale_factor=scaler, has_dropout=True, use_aspp=use_aspp) # .cuda()
    else:
        net = None
    return net