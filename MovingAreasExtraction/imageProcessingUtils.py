#########################################################
#                  LIBRARIES IMPORTS                    #
#########################################################

import os
from osgeo import gdal
import rioxarray as rio
from rioxarray.merge import merge_arrays


#########################################################
#                  SEGMENTATION UTILS                   #
#########################################################

def getSegFromShape(path):
    """
    Transform shapefile to segmentation image

    @input path :           Path to the shapefile to consider

    @return :               Segmentation image
    """

    #Create a temporary directory
    if not os.path.exists("./tempDir") :    
        os.mkdir("./tempDir")
    
    #Open shapefile
    shpDF = gdal.OpenEx(path)

    #Transform shapefile in masks
    NoData_value = 0
    pixel_size = 1e-4
    gdal.Rasterize("./tempDir/pilou.tif", shpDF, attribute = "id",format='GTIFF', outputType=gdal.GDT_UInt16, creationOptions=["COMPRESS=DEFLATE"], noData=NoData_value, initValues=NoData_value, xRes=pixel_size, yRes=-pixel_size, allTouched=True)

    #Read the created masks
    dataSeg = rio.open_rasterio("./tempDir/pilou.tif")
    
    return dataSeg



def adaptSeg(im, seg):
    """
    Adapt the segmentation based on the resolution of the image through linear interpolation

    @input im :         Image with the willing resolution
    @input seg :        Segmentation to be resampled

    @return :           Segmentation with the corresponding resolution
    """
    SegFinal = seg.interp(y = im.coords["y"].values, x = im.coords["x"].values, method = "nearest")
    SegFinal = SegFinal.fillna(0)
    
    return SegFinal



#########################################################
#                     IMAGE UTILS                       #
#########################################################

def combineSwath(data1, data2):
    """
    Combine swaths

    @input data1, data2 :       Images to be combined

    @return :                   Combined image
    """
    dataComb = merge_arrays([data1,data2])
    return dataComb


def cropImBasedOnSeg(im, seg):
    """
    Crop the image depending on the limits of the segmentation

    @input im :         Image to be croped
    @input seg :        Segmentation used as reference

    @return :           Croped image
    """

    #Get bbox of the mask
    x0 = seg.coords["x"][0].values
    xfin = seg.coords["x"][-1].values
    y0 = seg.coords["y"][0].values
    yfin = seg.coords["y"][-1].values
    
    #Crop image based on mask coordinates
    dataCrop = im.loc[:,y0:yfin, x0:xfin]
    
    return dataCrop
