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
        # First, find the minimum value in the depth image
        # for the depth image, the closer the object is to the camera, the smaller
        # the value. Thus, we want to find the closest pixel
        minVal = np.min(image[np.nonzero(image)])
        
        # Now we want to mask off the background. We do this by
        # setting any pixel thats further away than 1500 units from
        # the closest pixel to zero
        image[image > 1500 + minVal] = 0.

        # After masking off the background, find the furthest distance in our ROI
        # (note, this could be minVal + 1500, but it could be smaller)
        maxVal = np.max(image[np.nonzero(image)])

        # Now we preform two operations at once, the first is to scale the ROi relitive to itself
        # as such, the closest pixel should be near zero, and the furthest near one.
        # Second, we raise this value to the fourth power, this is to help make minor differences
        # between pixels more distinct.
        # For example: the sign 'A' versus the sign 'S'
        # 'A' has the thumb closer to the camera than in 'S', but the difference is thousanths of units 
        new_img = (image / maxVal) ** 4

        # Any small value (arbitrarily defined as smaller than 0.001, typically in the range ~E-5) is the 
        # result of floating point errors with the masked off pixels (I beleive)
        # Set these background pizels as 1, the furthest away in our range
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
