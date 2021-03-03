###############
#  others
import time
import string
import random


############### 
# for UI
import PIL # pip install Pillow
from PIL import Image,ImageTk
import pytesseract #     pip install pytesseract

from tkinter import * 
###############

###############
# for openCV and realsense  
import pyrealsense2 as rs
import numpy as np
import cv2
###############


# UI's TK window
root = Tk()
root.bind('<Escape>', lambda e: root.quit())
lmain = Label(root)
lmain.pack()
root.title('UI features demo') 


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
def show_depth():
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # Stack both images horizontally
   # images = np.hstack((color_image, depth_colormap))

    # Show images

    depthImgTk = ImageTk.PhotoImage(image=PIL.Image.fromarray(depth_image))
    lmain.imgtk = depthImgTk
    lmain.configure(image=depthImgTk)
    lmain.after(10, show_depth)
show_depth()



#labels and buttons
var = StringVar()

def random_sentence():
    char_lists = string.ascii_lowercase
    randomText = ''.join(random.choice(char_lists) for _ in range(10))
    randomText = 'predictions: ' + randomText
    var.set(randomText)

def clearSentence():
    var.set("")
random_sentence()
l2 = Label(root,  textvariable = var )
l2.pack()


bt2 = Button(root, text="clear", command=clearSentence)
bt2.pack()


#########
# menu demo
menubar = Menu(root) 
# Adding File Menu and commands 
file = Menu(menubar, tearoff = 0) 
menubar.add_cascade(label ='File', menu = file) 
file.add_command(label ='import model', command = None) 
file.add_command(label ='Save', command = None) 
file.add_command(label ='Save as', command = None) 
file.add_separator() 
file.add_command(label ='Exit', command = root.destroy) 

# Adding Help Menu 
help_ = Menu(menubar, tearoff = 0) 
menubar.add_cascade(label ='Help', menu = help_) 
help_.add_command(label ='Help info', command = None) 
help_.add_command(label ='Demo', command = None) 
help_.add_command(label ='Contact', command = None) 
help_.add_separator() 
help_.add_command(label ='About ASL recgonition', command = None) 
  

# display Menu 
root.config(menu = menubar) 
#############

root.mainloop() # start the window loop
