from .blocks import PlainBlock, ResidualBlock
from .unet import UNet, MultiEncoderUNet
from monai.networks.nets import UNETR


block_dict = {
    'plain': PlainBlock,
    'res': ResidualBlock
}


def get_unet(args):
    kwargs = {
        "input_channels"   : args.in_ch,
        "output_classes"   : args.num_classes,
        "channels_list"    : args.channels_list,
        "deep_supervision" : True, # args.deep_supervision,
        "ds_layer"         : 4, # args.ds_layer,
        "kernel_size"      : args.kernel_size,
        "dropout_prob"     : args.dropout_prob,
        "norm_key"         : args.norm,
        "block"            : block_dict[args.block],
    }
    
    if args.model == 'unet':
        return UNet(**kwargs)
    elif args.model == 'multiencoder_unet':
        return MultiEncoderUNet(**kwargs)
    elif args.model == 'unetr':
        return UNETR(
            spatial_dims=3,
            in_channels=args.in_ch,
            out_channels=args.num_classes,
            img_size=(args.patch_size, args.patch_size, args.patch_size),
            norm_name=args.norm,
            dropout_rate=args.dropout_prob,
        )
    else:
        raise NotImplementedError(args.model + " is not implemented.")