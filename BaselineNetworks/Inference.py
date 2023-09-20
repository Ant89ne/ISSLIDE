######################################################
#                    LIBRAIRIES                      #
######################################################

from metricsEval import getMetricsThreshold

from dataloader import MA_Truth_Dataset

import torch
from torch.utils.data import DataLoader

import numpy as np
import random
import matplotlib.pyplot as plt

from tqdm import tqdm

######################################################
#                 REPRODUCTIBILITY                   #
######################################################

seed = 1
torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

#########################################################
#                   PARAMS TO BE CHOSEN                 #
#########################################################

#Path parameters
path1 = ""
saveDir = ""
modelPath = ""

#########################################################
#                   DATASETS                            #
#########################################################

#Create evaluation dataset and dataloader
dataset_e = MA_Truth_Dataset(path1=path1,typedataset=1)
dataloader_e = DataLoader(dataset_e, batch_size=1, shuffle=True)

#########################################################
#                   DEEP MODEL                          #
#########################################################

#Load the pre-trained model
model = torch.load(modelPath, map_location="cpu")

#########################################################
#                   MAIN ROUTINE                        #
#########################################################

#Metrics initialization
nb = len(dataloader_e)
nbs = 100
precision, recall, rocFPR, rocTPR, dice, iou, hd = torch.zeros(nbs+1), torch.zeros(nbs+1), torch.zeros(nbs+1), torch.zeros(nbs+1), torch.zeros(nbs+1), torch.zeros(nbs+1), torch.zeros(nbs+1)
auc = 0

#Range of thresholds
s = [1- 1 / nbs * k for k in range(nbs+1)]

for d,l in tqdm(dataloader_e) :

    #Network prediction
    pred = model(d)

    #Metrics calculation
    auc, precisiono, recallo, diceo, iouo, hdo = getMetricsThreshold(pred["out"][0], l, nbs)

    #Save metrics values
    precision += precisiono
    recall += recallo
    dice += diceo
    iou += iouo
    hd += hdo
    
#Final metrics
recall = recall/nb
precision = precision/nb
dice = dice/nb
iou = iou/nb
hd = hd/nb

#F1_score calculation
f1_score = 2*(recall*precision)/(precision+recall+0.00000000000000000001)

#########################################################
#                   BEST METRICS                        #
#########################################################

#Extract best results (based on best F1-Score)
bestF1_index = np.argmax(f1_score)
bests = s[bestF1_index]
bestDice = dice[bestF1_index]
bestIoU = iou[bestF1_index]
bestHD = hd[bestF1_index]
bestF1 = f1_score[bestF1_index]

print(f"Best results for s = {bests}: \n F1-score: {bestF1} \n AUC: {auc} \n Dice: {bestDice} \n IoU: {bestIoU} \n Hausdorff Distance: {bestHD}")

#########################################################
#                   VISUALIZATION                       #
#########################################################

#Visualization
plt.subplot(212)
plt.plot(s,recall, '*')
plt.plot(s, precision, '*')
plt.plot(s, f1_score, '*')
plt.plot(s, dice, '*')
plt.plot(s,iou, '*')
plt.plot(s, (hd - np.min(hd))/(np.max(hd)- np.min(hd)))
plt.legend(["recall", "precision", "f1-score", "dice", "iou", "norm HD"])
plt.title("Metrics function of threshold")
plt.show()
