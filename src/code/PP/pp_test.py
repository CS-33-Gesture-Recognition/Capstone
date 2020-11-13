import sys
import os
import numpy as np
# set path to project so we can import our UTILS
sys.path.append(os.path.realpath('.'))
import UTIL.RSC_Wrapper as RSCW


if __name__ == "__main__":
    # create a PreProc object
    obj = RSCW.RSC()

    # a simple testing loop that will take a picture everytime enter is pressed
    # when any character is entered, the program will terminate
    while(True):
        # capture a new image
        obj.capture()

        # update the plot
        obj.display()

        # get the user inpuot to see if we will continue looping
        x = input("?")
        if (x != ''):
            break
