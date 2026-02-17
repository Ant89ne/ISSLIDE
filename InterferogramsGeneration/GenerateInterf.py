######################################################
#                    LIBRAIRIES                      #
######################################################

import os
import sys
import json
import argparse
#from tqdm import trange

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

for s in config["swaths"]:
    #Path toward input images
    pathInput = config["dataPaths"]["pathSplitSwath"] + f"Swath{s}/"
    #Path to the output folder to be created
    pathOutput = config["dataPaths"]["pathInterfero"] + f"Swath{s}/"
    #Path to the SNAP graph to be computed
    pathSNAPGraph = config["graphs"]["graphInterf"]
    #Path to the gpt executable
    pathGPT = config["gpt"]
    #Swath to be computed (one by one for memory issues)
    swath = s
    #Delay between two radar images on which to compute interferogram
    delay = config["delay"]

    ######################################################
    #                   INITIALIZATION                   #
    ######################################################

    # Find and sort SAR images to compute
    liste_imgs_org = [f for f in os.listdir(pathInput) if not f.endswith(".data") and f.startswith("S")]
    liste_imgs_org = date_sort(liste_imgs_org)

    #Check swath number
    if swath not in [1,2,3]:
        print("Error : wrong swath number")
        sys.exit(1)

    #Create output directory
    checkDir(pathOutput)

    #Number of successive images corresponding to the given delay
    deltat=int(delay)//6

    #Parameters for graphs configuration
    gptConfigParams = ["product1", "product2", "outputFile", "swath1", "swath2"]

    ######################################################
    #                   MAIN ROUTINE                     #
    ######################################################

    for i in range(0, len(liste_imgs_org)-deltat, deltat):
        #Configuration of user parameters
        outputName = f'{pathOutput}/ifg_IW{swath}_VV_{i}_{i+deltat}.dim'
        userParams = [pathInput + liste_imgs_org[i], pathInput + liste_imgs_org[i+deltat], outputName, f"IW{swath}", f"IW{swath}"]
        
        #Prepare command line
        comParams = ""
        for c, com in enumerate(gptConfigParams) :
            comParams += f"-P{com}={userParams[c]} "
        
        #Generate interferogram
        commandLine = f'{pathGPT} {pathSNAPGraph} {comParams}'
        os.system(commandLine)


    ######################################################
    #                SAVE CONFIGURATION                  #
    ######################################################

    #Path to the information file
    pathInfo = pathOutput + 'info.txt'

    #Open file and write data info
    f= open(pathInfo,"w+")
    f.write("Path configuration")
    f.write(f'Input path: {pathInput}\n')
    f.write(f'Graph path: {pathSNAPGraph}\n')
    f.write("Interferograms configuration")
    f.write(f'Swath number: {swath}\n')
    f.write(f"Delay: {delay}\n")
    f.write("Corresponding Images: ")
    for i, img in enumerate(liste_imgs_org):
        f.write(f"{i}: {img} ; ")
    f.close()
