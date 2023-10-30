######################################################
#                    LIBRAIRIES                      #
######################################################

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime

from dataloader import MA_Truth_Dataset
from Routines import training_routine, evaluation_routine
from utils import checkDir, visIms
from smallNets import SmallUNet, ResUNet, SepConvUNet
from adaptBigNets import BigNets

######################################################
#                 REPRODUCTIBILITY                   #
######################################################

seed = 1
torch.random.manual_seed(seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

#########################################################
#                   PARAMS TO BE CHOSEN                 #
#########################################################

#Path parameters
pathInput = ""
saveDir = ""


#Model parameters
modelName = "ResUNet"
pretrained = False

#Hyperparameters
epochsNb = 10      #Number of epochs
batchsize = 8       #Batch size
lr = 0.00001        #Learning rate

#########################################################
#                   AUTOMATIC PARAMS                    #
#########################################################

#Whether to use gpu or cpu
device = "cuda" if (torch.cuda.is_available()) else "cpu"      

#Create a specific saving directory
currDate = datetime.now()
saveDir += str(currDate).replace(' ', '_').replace(":","__") + "/"
checkDir(saveDir)


#########################################################
#                   DATASETS                            #
#########################################################

#Create training and evaluation datasets
dataset_t = MA_Truth_Dataset(path1=pathInput)
dataset_e = MA_Truth_Dataset(path1=pathInput, typedataset=1)

print(len(dataset_t), len(dataset_e))

#Create training and evaluation dataloaders
dataloader_t = DataLoader(dataset_t, batch_size=batchsize, shuffle=True, num_workers=10)
dataloader_e = DataLoader(dataset_e, batch_size=batchsize, shuffle=True, num_workers = 10)

#########################################################
#                   DEEP MODEL                          #
#########################################################

if modelName == "UNet":
    model = SmallUNet()
elif modelName == "ResUNet":
    model = ResUNet()
elif modelName == "SepUNet":
    model = SepConvUNet()
else :
    model = BigNets(modelName, pretrained)

model = model.to(device)

#########################################################
#              OPTIMIZATION PARAMETERS                  #
#########################################################

#Optimizer
optimizer = Adam(model.parameters(), lr=lr, betas=(0.5,0.999))

#Loss function
lossToUse =  nn.BCELoss()

#########################################################
#                   MAIN ROUTINE                        #
#########################################################

# Initializations
loss = []       #Saving loss    
mets = []       #Saving metrics

# Initial network performances
meanLoss_T, meanMet_T = evaluation_routine(dataloader_t, model, lossToUse, device)
meanLoss_E, meanMet_E = evaluation_routine(dataloader_e, model, lossToUse ,device)
loss.append([meanLoss_T, meanLoss_E])
mets.append([meanMet_T, meanMet_E])

# Training routine
for epoch in range(epochsNb):
    
    print(f"\n***********\n* Epoch {epoch} *\n***********\n")

    # Training
    meanLoss_T, meanMet_T = training_routine(dataloader_t, model, lossToUse, optimizer, device)
    print(f"\tTotal Loss : {meanLoss_T}")

    # Evaluation
    print("\nEvaluation")
    meanLoss_E, meanMet_E = evaluation_routine(dataloader_e, model, lossToUse, device)
    print(f"\tTotal Loss : {meanLoss_E}")

    # Save loss and metrics for visualization
    loss.append([meanLoss_T, meanLoss_E])
    mets.append([meanMet_T, meanMet_E])

    with torch.no_grad():
        
        #Save the model and image samples every 10 epochs
        if (epoch+1) % 10 == 0 :
            torch.save(model, f'{saveDir}/model{epoch+1}.pth')
            visIms(model, dataloader_e, saveDir, epoch+1, 20, device)

        #Save the loss and metrics curve every epochs
        if (epoch+1) % 1 == 0 :
            
            l = np.array(loss)
            m = np.array(mets)
            
            r = [k for k in range(len(l[:,0]))]

            #Visualization
            plt.figure(figsize=(20,10))

            #Visualize the loss function
            plt.subplot(332)
            plt.plot(r,l[:,0], '-r') 
            plt.plot(r,l[:,1], '-b')
            plt.title("Loss")

            #Visualize the loss function zoomed in
            if epoch + 1 > 5 :
                plt.subplot(131)
                plt.plot(r,l[:,0], '-r') 
                plt.plot(r,l[:,1], '-b')
                plt.axis([0, epoch+1, 0, np.sort(l[:,0])[-4]])
                plt.title("Zoom loss")

            #Visualize the Dice Score
            plt.subplot(333)
            plt.plot(r,m[:,0,0], '-r') 
            plt.plot(r,m[:,1,0], '-b')
            plt.title("Dice")

            #Visualize the Hausdorff Distance            
            plt.subplot(335)
            plt.plot(r,m[:,0,1], '-r') 
            plt.plot(r,m[:,1,1], '-b')
            plt.title("Hausdorff Distance")

            #Visualize the Precision
            plt.subplot(336)
            plt.plot(r,m[:,0,2], '-r') 
            plt.plot(r,m[:,1,2], '-b')
            plt.title("Intersection over Prediction (Precision)")

            #Visualize the Recall
            plt.subplot(338)
            plt.plot(r,m[:,0,3], '-r') 
            plt.plot(r,m[:,1,3], '-b')
            plt.title("Intersection over Truth (Recall)")

            #Visualize the IoU
            plt.subplot(339)
            plt.plot(r,m[:,0,4], '-r') 
            plt.plot(r,m[:,1,4], '-b')
            plt.title("IoU")

            #Save the plots
            plt.savefig(f'{saveDir}/loss{epoch+1}.png')
            plt.close()

