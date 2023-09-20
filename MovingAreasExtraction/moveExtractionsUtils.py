#########################################################
#                  LIBRARIES IMPORTS                    #
#########################################################

import cv2
import numpy as np
from skimage.feature import peak_local_max


#########################################################
#               COORDINATES MANIPULATION                #
#########################################################

def getCoords(seg):
    """
    Extract locations of moving areas

    @input seg :        Annotation of the moving areas

    @return :           Locations of the moving areas within a list (indices)
    """
    #Create the distance map on the mask
    dataDist = cv2.distanceTransform(np.array(seg[0]).astype(np.uint8),cv2.DIST_L2, 5)

    #Exttract local maxima to get location of moving areas
    maxis = peak_local_max(dataDist, 10)
    
    return maxis



def extractZoneFromCoords(im, y, x, size):
    """
    Crop an image based on geographic coordinates and size of the willing crop. Coordinates correspond to the center of the crop

    @input im :         Image to be cropped
    @input y, x :       Geographic coordinates of the center of the willing crop
    @input size :       Half-size of the willing crop

    @return :           Cropped image   
    """
    #Resolution of the image
    stepY = im.coords["y"][1]-im.coords["y"][0]
    stepX = im.coords["x"][1]-im.coords["x"][0]

    #Coordinates of the top left and bottom right corners
    ymin, ymax, xmin, xmax = y-size*stepY, y+size*stepY, x-size*stepX, x+size*stepX

    #Check if coordinates are out of bounds
    if ymin > im.coords["y"][0] :
        ymin = im.coords["y"][0]
        ymax = ymin + (2*size)*stepY

    if ymax < im.coords["y"][-1] :
        ymax = im.coords["y"][-1]
        ymin = ymax - (2*size)*stepY

    if xmin < im.coords["x"][0] :
        xmin = im.coords["x"][0]
        xmax = xmin + (2*size-1)*stepX

    if xmax > im.coords["x"][-1] :
        xmax = im.coords["x"][-1]
        xmin = xmax - (2*size-1)*stepX

    #Extract croped image
    movingIm = im.loc[:,ymin:ymax, xmin:xmax]

    return movingIm
