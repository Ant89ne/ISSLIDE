import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

def checkDir(path):
    """
    Check if directory already exists and create it
    
    @input pat :        Path to be checked
    """

    #Check if the path exists
    if os.path.exists(path):
        #Remove the path if it exists
        shutil.rmtree(path)    
    #Recreate it empty
    os.mkdir(path)


def visIms(model, dataloader, saveDir, epoch, nbIms = 20, device = "cpu"):
    """
    Function used for image results visualization

    @input model:           Model to be evaluated
    @input dataloader:      Dataset to be visually evaluated
    @input saveDir:         Folder to save images
    @input epoch:           Number of training epochs
    @input nbIms:           Number of images to visualize (default = 20)
    @input device:          Device on which to compute the evaluation (default = cpu)
    """

    #Create the saving folder
    checkDir(saveDir + f'/Ims{epoch}/')

    #Extract the dataset to be evaluated
    dataset = dataloader.dataset

    for k in range(nbIms):
        #Extract an image to be computed
        d,l = dataset[k]
        d = d.unsqueeze(0)

        #Get a prediction
        pred = model(d.to(device))
        pred = pred["out"].cpu()

        #Visualization
        plt.figure(figsize = (20,10))

        #Input coherence map
        plt.subplot(231)
        plt.imshow(d[0,0,:,:], vmin = 0, vmax = 1)
        plt.title("Coherence map")
        #Input Cosine
        plt.subplot(232)
        plt.imshow(d[0,1,:,:], vmin = -1, vmax = 1)
        plt.title("Cosine")
        #Input Sine
        plt.subplot(233)
        plt.imshow(d[0,2,:,:], vmin = -1, vmax = 1)
        plt.title("Sine")
        #Phase
        plt.subplot(234)
        plt.imshow(np.arctan2(d[0,2,:,:], d[0,1,:,:]), vmin = -np.pi, vmax = np.pi)
        plt.title("Phase")
        #Ground Truth
        plt.subplot(235)
        plt.imshow(l, vmin = 0, vmax = 1)
        plt.title("Ground Truth")
        #Prediction
        plt.subplot(236)
        plt.imshow(pred[0,0,:,:], vmin = 0, vmax = 1)
        plt.title("Prediction map")
        
        #Save image
        plt.savefig(saveDir + f'/Ims{epoch}/Im{k}.png')
        plt.close()