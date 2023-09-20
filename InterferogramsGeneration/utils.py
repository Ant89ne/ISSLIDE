import os


def date_sort(listImgs):
    """
    Sort filenames in liste by date if date included in filename
    """

    a = zip([f[3:] for f in listImgs], listImgs)
    a = sorted(a)

    b = [f for _,f in a]

    return b    

def checkDir(path):
    """
    Check if directory already exists and create it
    
    @input path :        Path to be checked
    """

    #Check if the path exists
    if not os.path.exists(path):
        #Recreate it empty
        os.mkdir(path)
