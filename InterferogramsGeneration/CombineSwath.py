 ######################################################
#                    LIBRAIRIES                      #
######################################################

from utils import checkDir
import os

######################################################
#                USER PARAMETERS                     #
######################################################

#Path toward input images
pathInput = "/media/bralet/Elements/DataToProcess/DESC/TestData/S1Data/"
#Path to the output folder to be created
pathOutput = "/media/bralet/Elements/DataToProcess/DESC/TestData/Combined/"
#Path to the SNAP graph to be computed
pathSNAPGraph = "/home/bralet/Bureau/CleanedWorks/ISSLIDE/InterferogramsGeneration/graphs/MergeImages.xml"
#Path to the gpt executable
pathGPT = "/opt/snapSentinel/bin/gpt"

######################################################
#                   INITIALIZATION                   #
######################################################

# Find and sort SAR images to compute
# /!\ Ensure images to combine are named to be the one after the other
allFiles = os.listdir(pathInput)
allFiles.sort()

#Create output directory
checkDir(pathOutput)

#Parameters for graphs configuration
gptConfigParams = ["swathNb", "file1", "file2", "outputName"]

######################################################
#                   MAIN ROUTINE                     #
######################################################

for k in range(0, len(allFiles), 2):
    #Get the two images to be combined
    file1 = pathInput + allFiles[k]
    file2 = pathInput + allFiles[k+1]

    #Combine swaths one by one
    for s in range(1,4):

        #Create Swath specific folder
        checkDir(pathOutput + f"Swath{s}/")

        #Configuration of user parameters
        values = [s, file1, file2, pathOutput + f"Swath{s}/" + allFiles[k][:-4]+ "_Asm" + str(s) + ".dim"]
        
        #Prepare command line
        commandline = ""
        for i in range(len(gptConfigParams)) :
            commandline += "-P" + gptConfigParams[i] + "=" + str(values[i]) + ' '

        #Combine images with SNAP
        os.system(pathGPT + " " + pathSNAPGraph + " " + commandline)