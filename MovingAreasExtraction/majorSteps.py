#########################################################
#                  LIBRARIES IMPORTS                    #
#########################################################

import rioxarray as rio
import numpy as np

from imageProcessingUtils import getSegFromShape, cropImBasedOnSeg, combineSwath, adaptSeg
from moveExtractionsUtils import getCoords, extractZoneFromCoords

#########################################################
#            PREPARE IMAGES FOR PATCHING                #
#########################################################


def getImageReady(globalSegPath, segPath, imPath, im2Path = None):
    """
    Prepare images for further process (routine function)

    @input globalSegPath:   Path towards a global segmentation file (including all segmentations)
    @input segPath :        Path towards the segmentation file
    @input imPath :         Path towards the image file
    @input imp2Path :       (Optionnal) Path towards a second image file

    @return :               Segmentation ready
    @return :               Image ready
    @return :               Locations of moving areas
    @return :               Global segmentation ready
    """

    #Transform the shapefile into a mask image
    dataSeg = getSegFromShape(segPath)

    #Read the global shapeFile
    dataSegGlobal = rio.open_rasterio(globalSegPath)

    #Extract locations of moving areas from the global shapeFile to have the same area for every interferograms (for multi-temporal issues)
    maxis = getCoords(dataSegGlobal)
    
    #Convert to geographical coordinates
    x = dataSegGlobal.coords["x"][maxis[:,1]]
    y = dataSegGlobal.coords["y"][maxis[:,0]]
    maxis = maxis.astype("float32")
    maxis[:,1] = x
    maxis[:,0] = y

    #Open image
    dataIm = rio.open_rasterio(imPath)
    #Crop the image based on the segmentation - only around the zone of interest
    dataCrop = cropImBasedOnSeg(dataIm, dataSegGlobal)
    

    #Open the second image
    if im2Path :
        #Open the image
        dataIm2 = rio.open_rasterio(im2Path)
        #Crop the image based on the segmentation
        dataCrop2 = cropImBasedOnSeg(dataIm2, dataSegGlobal)
        #Combine image1 and 2
        dataCrop = combineSwath(dataCrop, dataCrop2)
    
    #Adapt the segmentations to the resolution of the image
    dataSeg = adaptSeg(dataCrop, dataSeg)
    dataSegGlobal = adaptSeg(dataCrop, dataSegGlobal)
    

    return dataSeg, dataCrop, maxis, dataSegGlobal


#########################################################
#                   EXTRACT MOVES                       #
#########################################################



def getMovingAreas(im, seg, outputPath, imsize = 100, maxis = []):
    """
    Save crops of the moving areas of the zone

    @input im :             Image where crop have to be taken
    @input seg :            Segmentation used for generating coordinates
    @input outputPath :     Path where we save images
    @input imsize :         Size of the patch to extract (default : 100)
    @input maxis :          (Optionnal) Provide coordinates, if not provided, calculated inside the function
    """

    #Generate coordinates of the moving areas if not provided
    if not len(maxis) :
        maxis = getCoords(seg)

    #Extract moving areas
    halfS = imsize/2
    stepsegY = seg.coords["y"][1]-seg.coords["y"][0]
    stepsegX = seg.coords["x"][1]-seg.coords["x"][0]

    #Extract moving areas and save images
    for k, coords in enumerate(maxis) :
        y,x = coords[0], coords[1]

        #Get the id of the current move based on the global segmentation map
        moveIdx = np.max((seg.loc[:,y-stepsegY:y+stepsegY,x-stepsegX:x+stepsegX]).values)

        #Crop the image based on coordinates
        movingIm = extractZoneFromCoords(im, y, x, halfS)
        
        s1,s2,s3 = movingIm.shape
        if s1 != 1 or s2 != 100 or s3 !=100 :
            print("Extracted Shape Error: ", movingIm.shape)
        
        #Save the moving zone image
        movingIm.rio.to_raster(outputPath + f"{moveIdx}_{k}.tif")
