import torch
from torch.utils.data import Dataset

import os

import rioxarray as rio

import random

class MA_Truth_Dataset(Dataset):
    def __init__(self, path1, percentage = 0.8, typedataset = 0):
        super(MA_Truth_Dataset,self).__init__()

        #Attribute initialization
        self.path1=path1
        self.percentage = percentage
        self.typedataset = typedataset

        #Collect images
        self.getImages1()

        #Keep images containing actual moves
        self.keepOnes()
        
        #Test image ready and extract image shape
        imgTestName = self.dictSegImages[self.keys[0]][0]
        imgTest = rio.open_rasterio(imgTestName).data
        self.sizeZone = imgTest.shape

    def keepOnes(self):
        """
        Function to extract only images with actual moves
        """

        #Read segmentation files in reverse order
        for s in range(len(self.segs), -1, -1) :
            segIm = rio.open_rasterio(self.seg[s]).data

            #Remove filename if no actual move in the segmentation
            if not 1 in segIm :
                self.segs.pop(s)
                self.imgs.pop(s)

    def getImages1(self):
        """
        Look for the images and segmentations files available
        """

        #Initialization
        self.imgs = []
        self.segs = []

        #Extract the delays available
        delays = [f for f in os.listdir(self.path1)]

        for d in delays : 
            delayPath = self.path1 + d + '/'

            #Extract zones availables
            zones = sorted([f for f in os.listdir(delayPath)])

            #Avoid Queyras zone for the training and reserve it for testing
            if self.typedataset : 
                zones = [z for z in zones if "Bottom" in z] 
            else : 
                zones = [z for z in zones if "Bottom" not in z]

            for z in zones :
                zoneDelayPath = delayPath + z + "/"

                #Extract the names of the interferograms available
                interferos = sorted([f for f in os.listdir(zoneDelayPath) if (f != "Segmentations" and "30Oct" not in f)])

                #Keep only interferograms from the given dataset type
                if self.typedataset : 
                    interferos = interferos[int(self.percentage*len(interferos)):]
                else : 
                    interferos = interferos[:int(self.percentage*len(interferos))]

                #Path to the segmentation folder
                pathSeg = zoneDelayPath + "Segmentations/"

                #List moves observed in each interferogram as well as its annotations
                for i in interferos:
                    interfZoneDelayPath = zoneDelayPath + i + "/phase/"
                    imgs = [interfZoneDelayPath + f for f in os.listdir(interfZoneDelayPath) if f.endswith(".tif")]
                    imgs.sort()

                    interfSegPath = pathSeg + i + '/'
                    segs = [ interfSegPath + f for f in os.listdir(interfSegPath) if f.endswith('.tif')]
                    segs.sort()
                
                #Add the moves to all the moves available for the dataset
                self.imgs += imgs
                self.segs += segs

                    
    def __len__(self):
        """
        Function giving the total size of the dataset
        """
        return len(self.imgs)
    
    def __getitem__(self, idx):
        """
        Function used to get a sample of the dataset

        @input idx:     Index of the sample to extract
        """

        #Extract the path of the image and its segmentation
        path_img = self.imgs[idx]
        path_seg = self.segs[idx]

        #Read the segmentation
        imgSeg = rio.open_rasterio(path_seg).data
        imgSeg = torch.Tensor(imgSeg)

        #Choose a random phase offset for augmentation purposes
        alphaPI = random.random()*2*torch.pi

        #Read the phase difference
        img = rio.open_rasterio(path_img).data
        img = torch.Tensor(img)
        img = img  + alphaPI        #Add the offset

        #Read the coherence map
        coherence = rio.open_rasterio(path_img.replace("phase", "coherence")).data
        coherence = torch.Tensor(coherence)

        #Create the final three channels image: [coherence, cosine, sine]
        img = torch.concatenate((coherence, torch.cos(img), torch.sin(img)), 0)

        return img,imgSeg[0].type(torch.int64)
 