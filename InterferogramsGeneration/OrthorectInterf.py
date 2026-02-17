######################################################
#                    LIBRAIRIES                      #
######################################################

import os
import sys
# from tqdm import trange
import json
import argparse

from utils import date_sort, checkDir

parser = argparse.ArgumentParser()
parser.add_argument("--config", type = str, required=True)
args = parser.parse_args()

######################################################
#               READ CONFIG FILE                     #
######################################################

with open(args.config) as json_file:
    config = json.load(json_file)
    json_file.close()


######################################################
#                USER PARAMETERS                     #
######################################################

#Path toward input images
pathInputGlob = config["dataPaths"]["pathInterfero"]
#Path to the output folder to be created
pathOutputGlob = config["dataPaths"]["pathOrtho"]
#Path to the SNAP graph to be computed
pathSNAPGraph = config["graphs"]["graphOrtho"]
#Path to the gpt executable
pathGPT = config["gpt"]

######################################################
#                   INITIALIZATION                   #
######################################################
for s in config["swaths"]:
    pathInput = pathInputGlob + f"/Swath{s}/"
    pathOutput = pathOutputGlob + f"/Swath{s}/"

    # Find and sort interferograms to orthorectify
    liste_imgs_org = [e for e in os.listdir(pathInput) if e.endswith(".dim")]
    liste_imgs_org.sort()

    #Create subfolders in the output directory
    outputSubfolders = [f'{pathOutput}/phase/',f'{pathOutput}/coherence/']
    checkDir(pathOutput)
    for k in outputSubfolders :
        checkDir(k)

    #SNAP gpt necessary parameters
    gptConfigParams = ["product1", "extension", "outputPhase", "outputCoh"]

    ######################################################
    #                   MAIN ROUTINE                     #
    ######################################################

    for i in range(0, len(liste_imgs_org)):

        #Configuration of user parameters
        extensions = ["_pha", "_coh"]
        userParams = [pathInput + liste_imgs_org[i]]

        #Get band name within the .data folder (may be different from the actual filename)
        bands = [k for k in os.listdir(pathInput + liste_imgs_org[i][:-3] + "data") if k.startswith("coh_")][0]
        bName = bands[bands.find('_')+1:bands.rfind('.')]
        userParams += [bName]

        for k, ext in enumerate(extensions) :
            outFilePath = outputSubfolders[k] + liste_imgs_org[i][:-4]
            userParams.append(outFilePath + ext + ".tif")

        #Prepare command line
        comParams = ""
        for c, com in enumerate(gptConfigParams) :
            comParams += f"-P{com}={userParams[c]} "
        
        #Orthorectify Image
        commandLine = f'{pathGPT} {pathSNAPGraph} {comParams}'
        print(commandLine)
        os.system(commandLine)

    #Copy information path to output dir if info file exists
    if os.path.isfile(pathInput + "info.txt"):
        os.system(f"cp {pathInput}info.txt {pathOutput}info.txt")
