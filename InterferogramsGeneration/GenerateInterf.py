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
#Swath to be computed (one by one for memory issues)
swath = 1
#Delay between two radar images on which to compute interferogram
delay = 6

######################################################
#                   INITIALIZATION                   #
######################################################

# Find and sort SAR images to compute
liste_imgs_org = [f for f in os.listdir(pathInput) if not f.endswith(".data")]
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

for i in trange(0, len(liste_imgs_org)-deltat):
    #Configuration of user parameters
    outputName = f'{pathOutput}/ifg_IW{swath}_VV_{i}_{i+deltat}.dim'
    userParams = [liste_imgs_org[i], liste_imgs_org[i+deltat], outputName, str(swath), str(swath)]
    
    #Prepare command line
    comParams = ""
    for c, com in enumerate(gptConfigParams) :
        comParams += f"-P{com}={userParams[c]} "
    
    #Generate interferogram
    commandLine = f'{pathGPT} {pathSNAPGraph} comParams'
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
f.write("Corresponding Images")
for i, img in enumerate(liste_imgs_org):
    f.write(f"{i}: {img}")
f.close()
