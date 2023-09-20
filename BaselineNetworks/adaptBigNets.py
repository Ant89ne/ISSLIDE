import torch
import torch.nn as nn

from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50, FCN_ResNet50_Weights, DeepLabV3_ResNet50_Weights


class BigNets(nn.Module):
    """
    Class to adapt network from the literature
    """

    def __init__(self, netType, pretrained, classes = 1, sigmoid = True) -> None:
        """
        Initialization

        @input netType:         (str) Type of the network to test : FCN or DeepLabV3 allowed
        @input pretrained:      (boolean) Whether to use pretrained weights or not
        @input classes:         (int) Number of classes to be found (default = 1)
        @input sigmoid:         (boolean) Whether to use a sigmoid activation at the end of the network or not (default = True)
        """

        super(BigNets, self).__init__()

        #Discriminate the network to create
        if netType == "FCN" :
            #Discriminate whether to use pretrained weights
            if pretrained :
                self.model = fcn_resnet50(weights = FCN_ResNet50_Weights.DEFAULT)
                #Adapt last layer for the targeted number of classes
                self.model.classifier[-1] = nn.Conv2d(512,classes,1,1)
            else :
                self.model = fcn_resnet50(num_classes = classes)
        
        elif netType == "DeepLabV3" :
            #Discriminate whether to use pretrained weights
            if pretrained :
                self.model = deeplabv3_resnet50(weights = DeepLabV3_ResNet50_Weights.DEFAULT)
                #Adapt last layer for the targeted number of classes
                self.model.classifier[-1] = nn.Conv2d(256,classes,1,1)
                self.model.aux_classifier[-1] = nn.Conv2d(256,classes,1,1)
            else :
                self.model = deeplabv3_resnet50(num_classes = classes)

        #Sigmoid layer to be used if required
        self.sig = sigmoid

    def forward(self, x):
        """
        Forward pass of the block

        @input x :      Data to be forwarded
        """
        
        #Forward the model from the literature
        retInit = self.model(x)

        #Forward through a sigmoid layer if required
        if self.sig :
            ret = nn.Sigmoid()(retInit["out"])
        else :
            ret = retInit       

        return {"out" : ret}  