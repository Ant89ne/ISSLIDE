import os

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

######################################################
#                   INITIALIZATION                   #
######################################################

if not os.path.isdir(pathOutput):
    os.mkdir(pathInput)

names = ["inputFile", "outputFile"]
allFiles = os.listdir(pathInput)
allFiles.sort()

######################################################
#                   MAIN ROUTINE                     #
######################################################

for k in range(len(allFiles)):
    file1 = pathInput + allFiles[k]
    for s in range(1,4):
        values = [file1, pathOutput + allFiles[k][:-4] + ".dim"]
        commandline = " "

        for i in range(len(names)) :
            commandline += "-P" + names[i] + "=" + str(values[i]) + ' '

        print(commandline)

        os.system(pathGPT + " " + pathSNAPGraph + commandline)




