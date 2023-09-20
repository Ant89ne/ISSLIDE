from tqdm import tqdm
from metricsEval import getMetrics
import numpy as np
import torch

#########################################################
#                   TRAINING ROUTINE                    #
#########################################################

def training_routine(dataloader, model, loss, optimizer, device):
    """
    Function used to compute a whole epoch of training

    @input dataloader:      Dataloader to be used to train the network
    @input model:           Model to be trained
    @input loss:            Loss used to train the network
    @input optimizer:       Optimizer to use to train the network
    @input device:          Device on which run the training (cpu or cuda)

    @return :               Mean loss and metrics computed during the training
    """
    # Set model to training configuration
    model.train()

    #Initialization
    allmets = np.zeros(5)
    meanLoss = 0

    #Number of iteration on the dataloader
    s = len(dataloader)

    for k in tqdm(dataloader):
        #Set gradients to 0
        optimizer.zero_grad()

        # Send data to device
        inputImgs = k[0].to(device)
        segms = k[1].to(device)

        # Forward pass
        res = model(inputImgs)["out"]
        
        #Loss computation
        totalLoss = loss(res, segms.unsqueeze(1).type(torch.float))

        # Backward pass
        totalLoss.backward()
        
        # Optimization
        optimizer.step()

        # Metrics computations
        met = np.zeros(5)
        for ii, ima in enumerate(res) :
            met += getMetrics(ima, segms[ii].unsqueeze(1), device)
        
        #Save metrics and loss
        allmets += met/ii
        meanLoss += totalLoss.item()

    return meanLoss/s, allmets/s 

#########################################################
#                  EVALUATION ROUTINE                   #
#########################################################

def evaluation_routine(dataloader, model, losses, device):
    """
    Function to use for the evaluation of the performance of the network on a second dataset

    @input dataloader:      Dataloader to be used to train the network
    @input model:           Model to be trained
    @input loss:            Loss used to train the network
    @input device:          Device on which run the training (cpu or cuda)

    @return :               Mean loss and metrics computed during the evaluation
    """

    #Set the model to evaluation mode
    model.eval()
    
    #Initialization
    allmets = np.zeros(5)
    meanLoss = 0

    #Number of iterations on the dataaset
    s = len(dataloader)

    for k in tqdm(dataloader):

        # Send data to device
        inputImgs = k[0].to(device)
        segms = k[1].to(device)
        
        # Forward pass
        res = model(inputImgs)["out"]

        # Loss computation
        totalLoss = losses(res, segms.unsqueeze(1).type(torch.float))

        # Metrics calculations        
        met = np.zeros(5)
        for ii, ima in enumerate(res) :
            met += getMetrics(ima, segms[ii].unsqueeze(1), device)
        
        # Save
        allmets += met/ii
        meanLoss += totalLoss.item()

    return meanLoss/s, allmets/s