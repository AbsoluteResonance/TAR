
import torch
import torch.nn as nn
from torchvision import models

# This function creates a double convolutional layer block for the U-Net architecture
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        # First convolutional layer with input channels and output channels, kernel size of 3x3 and padding of 1
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        # ReLU activation function applied to the output of the first convolutional layer in place
        nn.ReLU(inplace=True),
        # Second convolutional layer with output channels, kernel size of 3x3 and padding of 1
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        # ReLU activation function applied to the output of the second convolutional layer in place
        nn.ReLU(inplace=True))


# Combines a 2D convolutional layer and a ReLU activation function.
class ConvolutionalReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        
        self.convolutional_relu_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding=padding),
            # inplace = True argument is used to modify the input tensor in place rather than creating a new one, which can save memory.
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        x = self.convolutional_relu_layer(x)
        return x
    
# Extracts features from input images and ultimately produces images with higher resolutions
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # This layer creates half as many output channels as input channels.
        self.conv1 = ConvolutionalReLU(in_channels, in_channels // 4, 1, 0)
        # This layer increases the resolution of the input images by 2x.
        self.deconv = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                         stride = 2, padding = 1, output_padding=0)
        self.conv2 = ConvolutionalReLU(in_channels // 4, out_channels, 1, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)
        
        return x
    
class UNetWithResNeXt50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        self.pretrained_model = models.resnext50_32x4d(pretrained=True)
        self.base_layers = list(self.pretrained_model.children())
        filters = [4*64, 4*128, 4*256, 4*512]
        
        # Down - sampling
        self.encoder0 = nn.Sequential(*self.base_layers[:3])
        self.encoder1 = nn.Sequential(*self.base_layers[4])
        self.encoder2 = nn.Sequential(*self.base_layers[5])
        self.encoder3 = nn.Sequential(*self.base_layers[6])
        self.encoder4 = nn.Sequential(*self.base_layers[7])
        
        # Up - sampling
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0]) 
        
        # Final Classifier
        self.last_conv0 = ConvolutionalReLU(256, 128, 3, 1)
        self.last_conv1 = nn.Conv2d(128, n_classes, kernel_size=3, padding=1)
        
    
    def forward(self, x):
        # Down - sampling
        # Reducing its spatial dimensions and increasing the number of feature maps.
        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Up sampling + skip connections
        # Performs up-sampling on the feature maps and combines them with the feature maps from corresponding encoder layer through skip connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # final classifier
        out = self.last_conv0(d1)
        out = self.last_conv1(out)
        out = torch.sigmoid(out)
        
        return out
if __name__=="__main__":
    # Initialize the U-Net architecture with 1 output class and move the model to the selected device.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Initialize the Unet with ResNeXt-50 Backbone model
    unet_rx50 = UNetWithResNeXt50(n_classes=1).to(device)

    # Pass a random tensor of shape (1,3,256,256) as an input to the UNet with ResNeXt-50 model, this tensor is also moved to the selected device
    output = unet_rx50(torch.randn(1,3,256,256).to(device))

    # Print the shape of the output of the UNet with ResNeXt-50 model, it should be (1,1,256,256)
    print(output.shape)