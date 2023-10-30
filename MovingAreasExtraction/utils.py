import os
import shutil

def checkDir(path, rmExist = True):
    """
    Check if directory already exists and create it
    
    @input pat :        Path to be checked
    """

    #Check if the path exists
    if os.path.exists(path) and rmExist:
        #Remove the path if it exists
        shutil.rmtree(path)    
    elif os.path.exists(path) and not rmExist:
        return
    #Recreate it empty
    os.mkdir(path)
