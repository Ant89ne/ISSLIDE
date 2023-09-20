######################################################
#                    LIBRAIRIES                      #
######################################################

import os
import sys
from tqdm import trange

from utils import date_sort, checkDir

######################################################
#                USER PARAMETERS                     #
######################################################

#Path toward input images
pathInput = ""
#Path to the output folder to be created
pathOutput = ""
#Path to the SNAP graph to be computed
pathSNAPGraph = ""
#Path to the gpt executable
pathGPT = ""

######################################################
#                   INITIALIZATION                   #
######################################################

# Find and sort interferograms to orthorectify
liste_imgs_org = [e for e in os.listdir(pathInput) if e.endswith(".dim")]
liste_imgs_org.sort()

#Create subfolders in the output directory
outputSubfolders = [f'{pathOutput}/phase/',f'{pathOutput}/coherence/']
checkDir(pathOutput)
for k in outputSubfolders :
    checkDir(k)

#SNAP gpt necessary parameters
gptConfigParams = ["product1", "outputPhase", "outputCoh"]

######################################################
#                   MAIN ROUTINE                     #
######################################################

for i in range(0, len(liste_imgs_org)):

    #Configuration of user parameters
    extensions = ["_pha", "_coh"]
    userParams = [liste_imgs_org[i]]
    for k, ext in enumerate(extensions) :
        outFilePath = outputSubfolders[k] + liste_imgs_org[i][:-4]
        userParams.append(outFilePath + ext + ".tif")

    #Prepare command line
    comParams = ""
    for c, com in enumerate(gptConfigParams) :
        comParams += f"-P{com}={userParams[c]} "
    
    #Generate interferogram
    commandLine = f'{pathGPT} {pathSNAPGraph} comParams'
    os.system(commandLine)

#Copy information path to output dir if info file exists
if os.path.isfile(pathInput + "info.txt"):
    os.system(f"cp {pathInput}info.txt {pathOutput}info.txt")
