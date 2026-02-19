 ######################################################
#                    LIBRAIRIES                      #
######################################################

from utils import checkDir
import os
import json
import argparse

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
pathInput = config["dataPaths"]["pathInput"]
#Path to the output folder to be created
pathOutput = config["dataPaths"]["pathSplitSwath"]
#Path to the SNAP graph to be computed
pathSNAPGraph = config["graphs"]["graphSplit"]
#Path to the gpt executable
pathGPT = config["gpt"]

######################################################
#                   INITIALIZATION                   #
######################################################

# Find SAR images to compute
allFiles = [k for k in os.listdir(pathInput) if k.startswith("S")]

#Create output directory
checkDir(pathOutput)

#Parameters for graphs configuration
gptConfigParams = ["swathNb", "file1", "outputName"]

######################################################
#                   MAIN ROUTINE                     #
######################################################

for k in range(0, len(allFiles)):
    #Get the two images to be combined
    file1 = os.path.join(pathInput, allFiles[k])

    #Combine swaths one by one
    for s in config["swaths"]:

        #Create Swath specific folder
        checkDir(os.path.join(pathOutput, f"SW{s}"))

        #Configuration of user parameters
        values = [f"IW{s}", file1, os.path.join(pathOutput, f"SW{s}", allFiles[k][:allFiles[k].rfind('.')]+ "_IW" + str(s) + ".dim")]
        
        #Prepare command line
        commandline = ""
        for i in range(len(gptConfigParams)) :
            commandline += "-P" + gptConfigParams[i] + "=" + str(values[i]) + ' '

        #Combine images with SNAP
        os.system(pathGPT + " " + pathSNAPGraph + " " + commandline)