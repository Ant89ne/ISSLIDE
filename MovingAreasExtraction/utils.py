import os
import shutil

def checkDir(path):
    """
    Check if directory already exists and create it
    
    @input pat :        Path to be checked
    """

    #Check if the path exists
    if os.path.exists(path):
        #Remove the path if it exists
        shutil.rmtree(path)    
    #Recreate it empty
    os.mkdir(path)
