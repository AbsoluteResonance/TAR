
import torch
import torch.nn as nn

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

class ConvReluUpSample(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False) :
        super().__init__()
        self.upsample = upsample
        self.make_upsample = nn.Upsample(scale_factor=2, mode="bilinear",align_corners=True)
        # Create a 2D convolutional layer with the specified number of input and output channels, kernel size of 3x3, stride of 1, and padding of 1. 
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size= (3,3),
                      stride = 1,
                      padding = 1,
                      bias = False),
            # Apply group normalization to the output of the convolutional layer
            nn.GroupNorm(num_groups = 32, 
                         num_channels = out_channels),
            # Apply ReLU activation function to the output of the group normalization
            nn.ReLU(inplace = True)
        )
    # Defines the forward method of the class, which takes an input tensor "x", and performs the convolutional block operations on it
    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            # Apply the upsampling operation to the output of the convolutional block.
            x = self.make_upsample(x)
        return x
    

class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()
        # This block is used to perform the convolution operation followed by ReLU activation and upsampling.
        blocks = [ConvReluUpSample(in_channels=in_channels,
                                   out_channels=out_channels,
                                   upsample=bool(n_upsamples))]
        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                #  This allows for multiple up-sampling operations to be performed in the block.
                blocks.append(ConvReluUpSample(in_channels=in_channels,
                                   out_channels=out_channels,
                                   upsample=True))
        
        self.block = nn.Sequential(*blocks)
    
    # Apply the input through the sequential block of convolutional layers, and return the output of the block.
    def forward(self,x):
        return self.block(x)
    
class FPN(nn.Module):
    
    def __init__(self, n_classes=1,
                 pyramid_channels = 256,
                 segmentation_channels = 256):
        super().__init__()
        
        # Bottom-up layers
        self.conv_bottomup1 = double_conv(3, 64)
        self.conv_bottomup2 = double_conv(64, 128)
        self.conv_bottomup3 = double_conv(128, 256)
        self.conv_bottomup4 = double_conv(256, 512)
        self.conv_bottomup5 = double_conv(512, 1024)
        self.maxpool = nn.MaxPool2d(2)
        
        # Top-down layer
        self.topdown = nn.Conv2d(1024, 256, kernel_size=1,
                                  stride=1, padding=0)
        
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # Lateral layers
        self.lateral1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.lateral2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.lateral3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        
        # Segmentation block layers
        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(in_channels=pyramid_channels,
                              out_channels=segmentation_channels,
                              n_upsamples=n_upsamples) for n_upsamples in [0, 1, 2, 3]])
        
        # Last layer
        self.last_conv = nn.Conv2d(256, n_classes, kernel_size=1, stride=1, padding=0)
    
    
    def upsample_add(self, x, y):
        _,_,H,W = y.size()
        upsample = nn.Upsample(size=(H,W),
                               mode="bilinear",
                               align_corners=True)
        return upsample(x) + y
    

    def upsample(self, x, h, w):
        sample = nn.Upsample(size=(h,w),
                             mode="bilinear",
                             align_corners=True)
        return sample(x)
    
    
    def forward(self, x):
            
            # Bottom-up
            c1 = self.maxpool(self.conv_bottomup1(x))
            c2 = self.maxpool(self.conv_bottomup2(c1))
            c3 = self.maxpool(self.conv_bottomup3(c2))
            c4 = self.maxpool(self.conv_bottomup4(c3))
            c5 = self.maxpool(self.conv_bottomup5(c4)) 
            
            # Top-down
            p5 = self.topdown(c5) 
            p4 = self.upsample_add(p5, self.lateral1(c4)) 
            p3 = self.upsample_add(p4, self.lateral2(c3))
            p2 = self.upsample_add(p3, self.lateral3(c2)) 
            
            # Smooth
            p4 = self.smooth1(p4)
            p3 = self.smooth2(p3)
            p2 = self.smooth3(p2)
            
            # Segmentation
            _, _, h, w = p2.size()
            feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p2, p3, p4, p5])]
            
            out = self.upsample(self.last_conv(sum(feature_pyramid)), 4 * h, 4 * w)
            
            out = torch.sigmoid(out)
            return out

if __name__=="__main__":
    # Initialize the U-Net architecture with 1 output class and move the model to the selected device.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Initialize the FPN architecture model
    fpn = FPN().to(device)

    # Pass a random tensor of shape (1,3,256,256) as an input to the FPN model, this tensor is also moved to the selected device
    output = fpn(torch.randn(1, 3, 256, 256).to(device))

    # Print the shape of the output of the FPN model, it should be (1,1,256,256)
    print(output.shape)