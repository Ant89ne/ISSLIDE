#########################################################
#               LIBRARIES IMPORTS                       #
#########################################################

from FirstTests import checkDir, getImageReady, getMovingAreas

import os
from tqdm import tqdm

#########################################################
#               PARAMS TO BE CHOSEN                     #
#########################################################

#Path to the folder with all Interferograms
interfPath = ""
#Path to the folder with all Shapefiles
shpPathAll = ""
#Path to a pre-computed global shapefile
GlobshpPathAll = "" 
#Output folder
savePathOrig = ""

#########################################################
#                  INITIALIZATION                       #
#########################################################

#Delays considered
delays = [f for f in os.listdir(interfPath) if os.path.isdir(f)]

#Individual shapefiles
shpFiles = []
for dd in [ d1 for d1 in os.listdir(shpPathAll) if os.path.isdir(shpPathAll + d1) and "AllInOne" not in d1]:                            #Across the delays
    for ff in [d2 for d2 in os.listdir(shpPathAll+dd+'/') if os.path.isdir(shpPathAll+dd+'/'+d2) and "AllInOne" not in d2]:             #Across the interferograms
        shpFiles += [shpPathAll + dd + '/' + ff + '/' + f for f in os.listdir(shpPathAll + dd + '/' + ff + '/') if f.endswith(".shp")]  #Across the zones

#Global shapefiles for each zone
globalShp = [GlobshpPathAll + f for f in os.listdir(GlobshpPathAll) if f.endswith(".tif")]

#Create output directory
checkDir(savePathOrig)

#########################################################
#                   MAIN ROUTINE                        #
#########################################################

for k in delays :
    #Available subfolders (typically phase and coherence)
    subfolds = [d for d in os.listdir(interfPath + k + "SW1") if os.path.isdir(interfPath + k + "SW1" + d)]

    #Select only shapefiles for the given delay
    delayShp = [f for f in shpFiles if k in f[f.rfind('/'):]]

    #Create an output sub-folder for the current delay
    delaySavePath = savePathOrig + k + '/'
    checkDir(delaySavePath)

    #Extract move in every shapefile
    for i, shp in enumerate(tqdm(delayShp)) :
        print( f"Processing {shp[shp.rfind('/')+1:-4]} shapeFile")
        
        #Get the zone of the current shapefile
        shpFileName = shp[shp.rfind('/')+1:-4]      #Extract shapefile filename
        zone = shpFileName[:shpFileName.find('_')]  #Extract the name of the zone
        
        #Create an output subfolder for the current zone (if not already existing)
        delayShpSavePath = delaySavePath + zone + '/' 
        checkDir(delayShpSavePath, False)

        #Identify the global shapefile corresponding to the given zone
        globalshapeFile = [g for g in globalShp if zone in g][0]

        #Extract the period of the interferogram used to create the shapefile
        first = shpFileName.rfind('_')+1
        second = shpFileName[:first-1].rfind('_') + 1
        dateIm = shpFileName[second:]

        #Apply extraction on each available band (typically coherence and phase)
        for subfold in subfolds :
            #Extract the interferogram corresponding to the current shapefile
            im = [f for f in os.listdir(interfPath + k + "SW1" + subfold) if dateIm in f][0] 

            #Create sub-folder in the output directory for the given period of time
            checkDir(delayShpSavePath + dateIm, False)
            #Create sub-folder in the output directory for the current band (coherence or phase)
            checkDir(delayShpSavePath + dateIm + '/' + subfold)
            
            #Final saving output directory for the current delay, zone, period, band
            path = delayShpSavePath + dateIm + '/' + subfold + '/'

            #Load image, rasterize the shapefile and extract coordinates of the moving areas
            seg, image, maxis, segGlob = getImageReady(globalshapeFile, shp, interfPath + k + "SW1" + subfold + '/' + im, interfPath + k + "SW2" + subfold + '/' + im.replace("IW1", "IW2"))
            seg = seg.astype("uint8")
            segGlob = segGlob.astype("uint8")
            
            #Extract moving areas on the current band
            getMovingAreas(image, segGlob, path, maxis = maxis)

        
        #Create output segmentation directory for the current delay and zone
        checkDir(delayShpSavePath + "Segmentations", False)
        
        #Create output segmentation folder for the current period
        checkDir(delayShpSavePath + "Segmentations/" + dateIm)

        #Extract segmentation of the movements extracted
        getMovingAreas(((seg>0)*1).astype("int8"), segGlob, delayShpSavePath + "Segmentations/" + dateIm + '/', maxis = maxis)

