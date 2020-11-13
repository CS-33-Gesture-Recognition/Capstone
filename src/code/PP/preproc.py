import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as color
import pyrealsense2 as rs
import numpy as np
import time


# make the matplotlib plots interactive, this allows them to
# be updated
plt.ion()


class PreProc:
    def __init__(self):
        # initilizer

        # Create the realsense pipeline object.
        # This pipeline is the interface with the camera
        self.pipeline = rs.pipeline()

        # Configure the camera, and connect
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # self.profile = self.pipeline.start(config)

        # Create an align object
        # rs.align allows us to perform alignment of depth
        #  frames to others frames
        # The "align_to" is the stream type to which we plan
        #  to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def fix_scale(self, image):
        minVal = np.min(image[np.nonzero(image)])
        
        image[image > 1500 + minVal] = 1

        maxVal = np.max(image[np.nonzero(image)])

        # image[image == 0] = maxVal

        # lets aim for a range of 0 -> 1000
        new_img = (image / maxVal) ** 4

        new_img[new_img < 0.001] = 1

        return new_img

    def capture(self):
        # activate the camera
        self.pipeline.start(self.config)

        # capture a depth image
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        # aligned_depth_frame is a 640x480 depth image
        aligned_depth_frame = aligned_frames.get_depth_frame() 

        # create the mono-color image as a np array
        img = np.asanyarray(aligned_depth_frame.get_data())

        # remove the background and save the image to the class
        self.depth_image = self.fix_scale(img)

        # deactivate the camera
        self.pipeline.stop()

    def update(self):
        # update the plot/image
        plt.imshow(self.depth_image, "gray_r")
        plt.show()

        # Matplotlib only updates the plots when the program is idle
        # thus, having the program preform a slight pause will ensure the 
        # plot is updated
        plt.pause(0.001)


if __name__ == "__main__":
    # create a PreProc object
    obj = PreProc()

    # a simple testing loop that will take a picture everytime enter is pressed
    # when any character is entered, the program will terminate
    while(True):
        # capture a new image
        obj.capture()

        # update the plot
        obj.update()

        # get the user inpuot to see if we will continue looping
        x = input("?")
        if (x != ''):
           break
