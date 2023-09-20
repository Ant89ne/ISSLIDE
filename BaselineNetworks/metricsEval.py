import torchmetrics as tmet
import torch

import numpy as np

import miseval

def getMetrics(segPred, segTruth):
    """
    Function to compute metrics based on a binarization with a threshold set to 0.5

    @input segPred:     Predicted segmentation
    @input segTruth:    Ground truth segmentation
    """

    with torch.no_grad():
        
        #Prepare ground truth for computation
        sT = segTruth.shape
        segTruth = segTruth.detach().cpu().squeeze()
        if sT[0] == 1 :
            segTruth = segTruth.unsqueeze(0)
        segTruth = np.array(segTruth)
        
        #Prepare prediction for computation
        sP = segPred.shape
        segPred = segPred.detach().cpu().squeeze()
        if sP[0] == 1 :
            segPred = segPred.unsqueeze(0)
        segPred = np.array(segPred > 0.5) 

        #Dice score computation
        diceScore = miseval.calc_DSC(segTruth,segPred)
        
        #Hausdorff distance computation
        hausdorffDist = miseval.calc_AverageHausdorffDistance(segTruth, segPred)

        #Precision and recall
        myeps = 0.0000000000001
        intersection = segPred * segTruth
        pixInter = np.sum(intersection)
        pixPred = np.sum(segPred)
        pixTruth = np.sum(segTruth)
        inter_pred = pixInter / (pixPred+myeps)     #Precision
        inter_truth = pixInter / (pixTruth+myeps)   #Recall

        #IoU computation
        iou = pixInter / (pixTruth + pixPred - pixInter+myeps)
        
    return np.array([diceScore, hausdorffDist, inter_pred, inter_truth, iou])


def getMetricsThreshold(segPred, segTruth, nbs = 1000):
    """
    Compute the metrics for different thresholds

    @input segPred:     Predicted segmentation
    @input segTruth:    Ground truth segmentation
    @input nbs:         Number of thresholds to compute
    """
    
    #List of thresholds to be tested
    s = [1- 1 / nbs * k for k in range(nbs+1)]

    # TODO check that !
    auc = tmet.AUROC(task = "binary", thresholds=s)

    finalauc = auc(torch.flatten(segPred), torch.flatten(segTruth) > 0)
    
    #Initialization of other metrics
    inter_pred = np.zeros(len(s))
    inter_truth = np.zeros(len(s))
    diceScore = np.zeros(len(s))
    iou = np.zeros(len(s))
    hd = np.zeros(len(s))
    for i, k in enumerate(s) :
        # Threshold the prediction for the given threshold
        pred = np.array(segPred>k)
        # Ensure the ground truth to be binary
        truth = np.array(segTruth > 0)

        #Compute precision and recall
        myeps = 0.0000000000001
        intersection = pred * truth
        pixInter = np.sum(intersection)
        pixPred = np.sum(pred)
        pixTruth = np.sum(truth)
        inter_pred[i] = pixInter / (pixPred+myeps)      #Precision
        inter_truth[i] =  pixInter / (pixTruth+myeps)   #Recall

        #Compute IoU
        iou[i] = pixInter / (pixTruth + pixPred - pixInter+myeps)

        #Compute Dice Score
        diceScore[i] = miseval.calc_DSC(truth,pred)

        #Compute hausdorff distance
        hd[i] = miseval.calc_AverageHausdorffDistance(truth, pred)



    return finalauc, inter_pred, inter_truth, diceScore, iou, hd
