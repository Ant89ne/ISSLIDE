#############################################################
#                    IMPORT LIBRAIRIES                      #
#############################################################

#Pytorch librairies
import torch
import torch.nn as nn

#############################################################
#                  SEPARABLE CONVOLUTIONS                   #
#############################################################

class SepConvBlock(nn.Module):
    """
    Class for the separable convolution block
    """

    def __init__(self, inputChannels, outputChannels, kernel, K=1) -> None:
        """
        Initialization

        @input inputChannels:       Number of channels as input
        @input outputChannels:      Number of willing output channels
        @input kernel:              Size of the convolutional kernels
        @input K:                   Number of spatial kernels to apply to each channel in the depthwise convolution
        """

        super(SepConvBlock, self).__init__()

        #Dephtwise convolution : each channel is convolved separately and independently (groups argument)
        self.depthwiseConv = nn.Conv2d(inputChannels, K*inputChannels, kernel, 1,1, groups=inputChannels, padding_mode="reflect")

        #Pointwise convolution : along the channel dimension        
        self.pointwise = nn.Conv2d(K*inputChannels, outputChannels, 1)


    def forward(self, x):
        """
        Forward pass of the block

        @input x :      Data to be forwarded
        """
        #Apply the convolutions to the input
        ret = self.depthwiseConv(x)
        ret = self.pointwise(ret)

        return ret        

#############################################################
#                    RESIDUAL BLOCK                         #
#############################################################

class ResBlock(nn.Module):
    """
    Class for the residual block
    """

    def __init__(self, inputChannels, kernel) -> None:
        """
        Initialization

        @input inputChannels:       Number of channels as input
        @input kernel:              Size of the convolutional kernels
        """

        super(ResBlock, self).__init__()

        self.layers = []
        for k in range(2):
            self.layers.append(nn.Conv2d(inputChannels, inputChannels, kernel, padding="same"))
            self.layers.append(nn.ReLU())
        
        self.layers = nn.Sequential(*self.layers)
        

    def forward(self, x):
        """
        Forward pass of the block

        @input x :      Data to be forwarded
        """
        ret = self.layers(x)

        #Apply the residual connection
        ret = ret + x

        return ret        

#############################################################
#                    CLASSICAL U-NET                        #
#############################################################

class SmallUNet(nn.Module):
    """
    Class for a siple U-Net network
    """
    def __init__(self, inputChannels = 3):
        """
        Initialization

        @input inputChannels:       Number of channels of the input image (default: 3)
        """

        super(SmallUNet, self).__init__()

        #Attibutes configuration
        self.inputChannels = inputChannels

        # Number of convolutional kernels for each layer
        kernels = [12,24,48]

        #First encoder convolutional block
        self.conv_0 = nn.Sequential(
            nn.Conv2d(self.inputChannels,kernels[0],3, padding="same"),
            nn.ReLU(),            
            nn.BatchNorm2d(kernels[0]),
            nn.Conv2d(kernels[0], kernels[0],3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(kernels[0], kernels[0],3, padding="same"),
            nn.ReLU()
        )

        #Second encoder concolutional block
        self.conv_1 = nn.Sequential(
            nn.Conv2d(kernels[0],kernels[1],3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(kernels[1]),
            nn.Conv2d(kernels[1], kernels[1],3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(kernels[1], kernels[1],3, padding="same"),
            nn.ReLU()
        )

        #Latent space processing
        self.latent = nn.Sequential(
            nn.Conv2d(kernels[1],kernels[2],3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(kernels[2]),
            nn.Conv2d(kernels[2], kernels[2],3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(kernels[2], kernels[2],3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(kernels[2],kernels[1],3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(kernels[1])
        )

        #First decoder convolutional block
        self.upconv_1 = nn.Sequential(
            nn.Conv2d(kernels[2], kernels[1],3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(kernels[1], kernels[1],3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(kernels[1], kernels[0],3, padding="same"),
            nn.BatchNorm2d(kernels[0]),
            nn.ReLU()
        )

        #Second decoder convolutional block
        self.upconv_0 = nn.Sequential(
            nn.Conv2d(kernels[1],kernels[0],3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(kernels[0], kernels[0],3, padding="same"),
            nn.BatchNorm2d(kernels[0])#,
            # nn.ReLU()
        )
        
        #Final classification layer
        self.finalConv = nn.Conv2d(kernels[0], 1, 3, padding="same")
        

    def forward(self, x):
        """
        Forward pass

        @input x :      Data to be forwarded
        """

        # Encoder pass
        ret0 = self.conv_0(x)
        ret1 = self.conv_1(nn.MaxPool2d(2)(ret0))
        lat = self.latent(nn.MaxPool2d(2)(ret1))

        #Low resolution skip connection
        skip1 = torch.cat((ret1, nn.Upsample(scale_factor=2)(lat)), dim=1 )
        upret1 = self.upconv_1(skip1)

        #High resolution skip connection
        skip0 = torch.cat((ret0, nn.Upsample(scale_factor=2)(upret1)), dim=1 )
        upret0 = self.upconv_0(skip0)

        #Final classification
        finalRet = nn.Sigmoid()(self.finalConv(upret0))

        return {"out" : finalRet}
    
#############################################################
#                    Residual  U-NET                        #
#############################################################

class ResUNet(nn.Module):
    """
    Class for the Residual U-Net
    """

    def __init__(self, inputChannels = 3):
        """
        Initialization
       
        @input inputChannels:       Number of channels of the input image (default: 3)
        """

        super(ResUNet, self).__init__()


        #Attributes configuration
        self.inputChannels = inputChannels

        #Number of convolutional kernels for each layer
        kernels = [12,24,48]

        #First encoder residual block
        self.conv_0 = nn.Sequential(
            nn.Conv2d(self.inputChannels,kernels[0],3, padding="same"),
            nn.ReLU(),            
            nn.BatchNorm2d(kernels[0]),
            ResBlock(kernels[0],3)
        )

        #second encoder residual block
        self.conv_1 = nn.Sequential(
            nn.Conv2d(kernels[0],kernels[1],3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(kernels[1]),
            ResBlock(kernels[1],3)
        )

        #Latent space
        self.latent = nn.Sequential(
            nn.Conv2d(kernels[1],kernels[2],3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(kernels[2]),
            ResBlock(kernels[2],3),
            nn.Conv2d(kernels[2],kernels[1],3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(kernels[1])
        )
        
        #First decoder convolutional block
        self.upconv_1 = nn.Sequential(
            nn.Conv2d(kernels[2], kernels[1],3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(kernels[1], kernels[1],3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(kernels[1], kernels[0],3, padding="same"),
            nn.BatchNorm2d(kernels[0]),
            nn.ReLU()
        )

        #Second decoder convolutional block
        self.upconv_0 = nn.Sequential(
            nn.Conv2d(kernels[1],kernels[0],3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(kernels[0], kernels[0],3, padding="same"),
            nn.BatchNorm2d(kernels[0])#,
        )
        
        #Final classification layer
        self.finalConv = nn.Conv2d(kernels[0], 1, 3, padding="same")
        

    def forward(self, x):
        """
        Forward pass

        @input x :      Data to be forwarded
        """

        #Encoder pass
        ret0 = self.conv_0(x)
        ret1 = self.conv_1(nn.MaxPool2d(2)(ret0))
        lat = self.latent(nn.MaxPool2d(2)(ret1))

        #Low resolution skip connection
        skip1 = torch.cat((ret1, nn.Upsample(scale_factor=2)(lat)), dim=1 )
        upret1 = self.upconv_1(skip1)

        #High resolution skip connection
        skip0 = torch.cat((ret0, nn.Upsample(scale_factor=2)(upret1)), dim=1 )
        upret0 = self.upconv_0(skip0)

        #Final classification
        finalRet = nn.Sigmoid()(self.finalConv(upret0))

        return {"out" : finalRet}
    
#############################################################
#                    Sep Conv  U-NET                        #
#############################################################

class SepConvUNet(nn.Module):
    """
    Class for the separable convolutions U-Net
    """

    def __init__(self, inputChannels = 3):
        """
        Initialization

        @input inputChannels:       Number of channels of the input image (default: 3)
        """

        super(SepConvUNet, self).__init__()

        #Attributes configuration
        self.inputChannels = inputChannels

        #Number of convolutional kernels for each layer
        kernels = [12,24,48]

        #First encoder separable convolutional block
        self.conv_0 = nn.Sequential(
            SepConvBlock(self.inputChannels, kernels[0], 3, 2),
            nn.ReLU(),            
            nn.BatchNorm2d(kernels[0]),
            SepConvBlock(kernels[0], kernels[0], 3, 2),
            nn.ReLU(),            
            SepConvBlock(kernels[0], kernels[0], 3, 2),
            nn.ReLU()   
        )

        #Second encoder separable convolutional block
        self.conv_1 = nn.Sequential(
            SepConvBlock(kernels[0], kernels[1], 3, 2),
            nn.ReLU(),            
            nn.BatchNorm2d(kernels[1]),
            SepConvBlock(kernels[1], kernels[1], 3, 2),
            nn.ReLU(),            
            SepConvBlock(kernels[1], kernels[1], 3, 2),
            nn.ReLU()
        )

        #Latent space
        self.latent = nn.Sequential(
            SepConvBlock(kernels[1], kernels[2], 3, 2),
            nn.ReLU(),            
            nn.BatchNorm2d(kernels[2]),
            SepConvBlock(kernels[2], kernels[2], 3, 2),
            nn.ReLU(),            
            SepConvBlock(kernels[2], kernels[2], 3, 2),
            nn.ReLU(),
            nn.Conv2d(kernels[2],kernels[1],3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(kernels[1])
        )

        #First decoder convolutional block
        self.upconv_1 = nn.Sequential(
            nn.Conv2d(kernels[2], kernels[1],3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(kernels[1], kernels[1],3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(kernels[1], kernels[0],3, padding="same"),
            nn.BatchNorm2d(kernels[0]),
            nn.ReLU()
        )

        #Second decoder convolutional block
        self.upconv_0 = nn.Sequential(
            nn.Conv2d(kernels[1],kernels[0],3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(kernels[0], kernels[0],3, padding="same"),
            nn.BatchNorm2d(kernels[0])
        )
        
        #Final classification layer
        self.finalConv = nn.Conv2d(kernels[0], 1, 3, padding="same")
        

    def forward(self, x):
        """
        Forward pass

        @input x :      Data to be forwarded
        """

        #Encoder pass
        ret0 = self.conv_0(x)
        ret1 = self.conv_1(nn.MaxPool2d(2)(ret0))
        lat = self.latent(nn.MaxPool2d(2)(ret1))

        #Low resolution skip connection
        skip1 = torch.cat((ret1, nn.Upsample(scale_factor=2)(lat)), dim=1 )
        upret1 = self.upconv_1(skip1)

        #High resolution skip connection
        skip0 = torch.cat((ret0, nn.Upsample(scale_factor=2)(upret1)), dim=1 )
        upret0 = self.upconv_0(skip0)

        #Final classification
        finalRet = nn.Sigmoid()(self.finalConv(upret0))

        return {"out" : finalRet}